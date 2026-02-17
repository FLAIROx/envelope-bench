import dataclasses
import time

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx

# Compat shims for older nnx API used by networks module
if not hasattr(nnx, "Pytree"):
    nnx.Pytree = nnx.Object
if not hasattr(nnx, "static"):
    nnx.static = lambda x: x
if not hasattr(nnx, "data"):
    nnx.data = lambda x: x

import envelope
from envelope.typing import PyTree
from ppo_vmap.logger import Logger
from ppo_vmap.networks import (
    DiscretePolicy,
    GaussianPolicy,
    ValueFunction,
    symexp,
    symlog,
)


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "envpool::CartPole-v1"
    total_timesteps: int = 1_000_000
    policy_lr: float = 0.0003
    policy_wd: float = 0.0001
    value_fn_lr: float = 0.0001
    value_wd: float = 0.0001
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    num_envs: int = 2000
    num_minibatches: int = 20
    num_epochs: int = 8
    num_steps: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_observations: bool = True
    normalize_rewards: bool = False
    seed: int = 0
    num_threads: int = 0

    # network arch
    activation: str = "tanh"
    layer_size: int = 256
    use_symlog: bool = False
    optimizer: str = "adamw"

    # logging
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-ppo"
    log_every: int = 131_072

    # fixed: envpool can't be vmapped
    num_runs: int = 1

    # learning rate schedule
    anneal_lr: bool = False

    # checkpointing
    num_checkpoints: int = 0


def make_env(args: Args):
    env_kwargs = {
        "batch_size": args.num_envs,
        "num_envs": args.num_envs,
        "seed": args.seed,
    }
    if args.num_threads > 0:
        env_kwargs["num_threads"] = args.num_threads

    env = envelope.create(args.env_name, env_kwargs=env_kwargs)
    env = envelope.ContinuousObservationWrapper(env)
    env = envelope.FlattenObservationWrapper(env)

    # Optionally wrap with normalization (these support BatchedSpace)
    if args.normalize_observations:
        env = envelope.ObservationNormalizationWrapper(env=env)
    if args.normalize_rewards:
        env = envelope.RewardNormalizationWrapper(env=env, discount=args.gamma)

    return env


class TrainState(nnx.Object):
    def __init__(self, args: Args):
        self.args = args  # non-array; stored in graphdef
        self.global_steps = jnp.array(0)
        self.run_idx = jnp.array(0)

        # Initialize environment
        env = make_env(args)
        self.vecenv = env  # FrozenPyTreeNode with static fields; stored in graphdef
        self.rngs = nnx.Rngs(args.seed)

        # Network init uses unbatched (single-env) spaces
        obs_space = env.observation_space.space
        act_space = env.action_space.space

        discrete = isinstance(act_space, envelope.Discrete)
        policy_cls = DiscretePolicy if discrete else GaussianPolicy
        self.policy = policy_cls(
            obs_space,
            act_space,
            self.rngs,
            activation=args.activation,
            layer_size=args.layer_size,
        )
        self.value_fn = ValueFunction(
            obs_space,
            self.rngs,
            activation=args.activation,
            layer_size=args.layer_size,
            use_symexp=args.use_symlog,
        )

        # Initialize optimizers
        num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
        total_opt_steps = num_updates * args.num_epochs * args.num_minibatches
        if args.anneal_lr:
            policy_lr = optax.linear_schedule(args.policy_lr, 0.0, total_opt_steps)
            value_lr = optax.linear_schedule(args.value_fn_lr, 0.0, total_opt_steps)
        else:
            policy_lr = args.policy_lr
            value_lr = args.value_fn_lr

        try:
            optimizer = getattr(optax, args.optimizer)
        except AttributeError:
            optimizer = getattr(optax.contrib, args.optimizer)

        policy_opt = optimizer(policy_lr, eps=1e-5, weight_decay=args.policy_wd)
        self.policy_optimizer = nnx.Optimizer(self.policy, policy_opt, wrt=nnx.Param)
        value_opt = optimizer(value_lr, eps=1e-5, weight_decay=args.value_wd)
        self.value_fn_optimizer = nnx.Optimizer(self.value_fn, value_opt, wrt=nnx.Param)

        # Initialize environment state and info
        env_state, env_info = self.vecenv.init(self.rngs())
        self.env_state = env_state
        self.env_info = env_info

        # Episode statistics tracking (shape: (num_envs,))
        self.ep_reward = jnp.zeros(args.num_envs)
        self.ep_length = jnp.zeros(args.num_envs)
        self.last_done_reward = jnp.full(args.num_envs, jnp.nan)
        self.last_done_length = jnp.full(args.num_envs, jnp.nan)


def shuffle_and_split(data: PyTree, num_minibatches: int, key: jax.Array):
    first_leaf = jax.tree.leaves(data)[0]
    num_steps, num_envs = first_leaf.shape[:2]
    batch_size = num_steps * num_envs
    permutation = jax.random.permutation(key, batch_size)

    def _shuffle_and_split(x):
        x = x.reshape((batch_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)


def collect_trajectories(ts: TrainState):
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_steps)
    def step_env(ts: TrainState):
        obs = ts.env_info.obs
        value = ts.value_fn(obs)

        pi = ts.policy(obs)
        action = pi.sample(seed=ts.rngs())

        env_state, env_info = ts.vecenv.step(ts.env_state, action)
        ts.global_steps += ts.args.num_envs
        ts.env_state = env_state

        # Update episode statistics
        ts.ep_reward = ts.ep_reward + env_info.reward
        ts.ep_length = ts.ep_length + 1
        done = env_info.terminated | env_info.truncated
        ts.last_done_reward = jnp.where(done, ts.ep_reward, ts.last_done_reward)
        ts.last_done_length = jnp.where(done, ts.ep_length, ts.last_done_length)
        ts.ep_reward = jnp.where(done, 0.0, ts.ep_reward)
        ts.ep_length = jnp.where(done, 0.0, ts.ep_length)

        ts.env_info = env_info

        out_info = env_info.update(
            obs=obs,
            value=value,
            action=action,
            log_prob=pi.log_prob(action),
            value_next=ts.value_fn(env_info.final.obs),
            last_done_reward=ts.last_done_reward,
            last_done_length=ts.last_done_length,
        )
        return ts, out_info

    ts, out_info = step_env(ts)
    return out_info


def calculate_gae(ts: TrainState, info):
    @nnx.scan(reverse=True)
    def gae_step(carry, transition):
        gae, next_value = carry
        done = transition.terminated | transition.truncated

        next_value = jnp.where(transition.truncated, transition.value_next, next_value)
        next_value = jnp.where(transition.terminated, 0, next_value)

        delta = transition.reward + ts.args.gamma * next_value - transition.value
        gae = delta + ts.args.gamma * ts.args.gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    # Get last value for bootstrapping
    done = ts.env_info.terminated | ts.env_info.truncated
    last_obs = jnp.where(done[:, None], ts.env_info.final.obs, ts.env_info.obs)
    last_value = ts.value_fn(last_obs)
    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = gae_step(init_carry, info)
    return advantages


def update_policy(ts: TrainState, batch):
    def normalize(x: jax.Array) -> jax.Array:
        return (x - x.mean()) / (x.std() + 1e-8)

    @nnx.value_and_grad(has_aux=True)
    def loss_fn(policy):
        pi = policy(batch.obs)
        log_prob = pi.log_prob(batch.action)
        entropy = pi.entropy().mean()

        log_ratio = log_prob - batch.log_prob
        ratio = jnp.exp(log_ratio)
        clip_ratio = jnp.clip(ratio, 1 - ts.args.epsilon, 1 + ts.args.epsilon)
        advantages = normalize(batch.advantages)

        surrogate1 = ratio * advantages
        surrogate2 = clip_ratio * advantages
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
        loss = policy_loss - ts.args.entropy_coef * entropy

        clip_frac = jnp.mean(ratio != clip_ratio)
        approx_kl = jnp.mean(((ratio - 1) - log_ratio))
        metrics = {
            "policy/clipped_surrogate_loss": policy_loss,
            "policy/entropy": entropy,
            "policy/clip_frac": clip_frac,
            "policy/approx_kl": approx_kl,
        }
        return loss, metrics

    (loss, loss_metrics), grads = loss_fn(ts.policy)
    ts.policy_optimizer.update(grads)
    _, state = nnx.split(ts.policy)
    param_norm = optax.global_norm(state)
    grad_norm = optax.global_norm(grads)
    return {
        "policy/loss": loss,
        "policy/grad_norm": grad_norm,
        "policy/param_norm": param_norm,
        **loss_metrics,
    }


def update_value_fn(ts: TrainState, batch):
    @nnx.value_and_grad(has_aux=True)
    def loss_fn(value_fn):
        targets = batch.value + batch.advantages
        raw_values = value_fn.raw(batch.obs)
        if ts.args.use_symlog:
            targets = symlog(targets)
        values = raw_values if not ts.args.use_symlog else symexp(raw_values)
        return 0.5 * jnp.mean((raw_values - targets) ** 2), values.mean()

    (loss, values), grads = loss_fn(ts.value_fn)
    ts.value_fn_optimizer.update(grads)
    grad_norm = optax.global_norm(grads)
    _, state = nnx.split(ts.value_fn)
    param_norm = optax.global_norm(state)
    return {
        "value/loss": loss,
        "value/grad_norm": grad_norm,
        "value/param_norm": param_norm,
        "value/mean_prediction": values,
    }


def update_epoch(ts: TrainState, minibatches):
    @nnx.scan
    def update_minibatch(ts: TrainState, batch):
        policy_info = update_policy(ts, batch)
        value_info = update_value_fn(ts, batch)
        return ts, {**policy_info, **value_info}

    _, loss_info = update_minibatch(ts, minibatches)
    return loss_info


def train_step(ts: TrainState):
    # Collect trajectories
    info = collect_trajectories(ts)

    # Compute advantages
    advantages = calculate_gae(ts, info)
    info = info.update(advantages=advantages)

    # Multiple epochs of updates
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_epochs)
    def update_epoch_scan(ts):
        minibatches = shuffle_and_split(info, ts.args.num_minibatches, ts.rngs())
        loss_info = update_epoch(ts, minibatches)
        return ts, loss_info

    ts, loss_infos = update_epoch_scan(ts)
    # take mean over minibatches but not epochs
    size = loss_infos["policy/clip_frac"].shape
    clip_fracs_per_epoch = loss_infos["policy/clip_frac"].mean(axis=1)
    epoch_clip_frac_dict = {
        f"policy/clip_frac_{i}": clip_fracs_per_epoch[i] for i in range(size[0])
    }
    loss_infos = jax.tree.map(jnp.mean, loss_infos)
    loss_infos = {**loss_infos, **epoch_clip_frac_dict}
    return info.update(loss_infos=loss_infos)


def make_block_fn(block_size: int, logger: Logger):
    """Create a training block function that runs block_size PPO updates."""

    @nnx.scan(in_axes=nnx.Carry, length=block_size)
    def train_block(ts: TrainState):
        out_info = train_step(ts)
        mean_return = out_info.last_done_reward.mean()
        std_return = out_info.last_done_reward.std()
        mean_episode_length = out_info.last_done_length.mean()
        std_episode_length = out_info.last_done_length.std()
        metrics = {
            "episode/return": mean_return,
            "episode/return_std": std_return,
            "episode/length": mean_episode_length,
            "episode/length_std": std_episode_length,
            **out_info.loss_infos,
        }
        jax.debug.callback(logger.log, ts.global_steps, ts.run_idx, metrics)
        return ts, mean_return

    return train_block


if __name__ == "__main__":
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    logger = Logger(args)

    # Compute block structure
    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update
    num_blocks = max(args.num_checkpoints, 1)
    block_size = num_updates // num_blocks
    assert block_size * num_blocks == num_updates, (
        f"num_updates ({num_updates}) must be divisible by num_blocks ({num_blocks})"
    )

    # Create train state (single run, no vmap/pmap)
    ts = TrainState(args)
    train_block = make_block_fn(block_size, logger)

    # AOT compile via split/merge pattern (nnx.jit doesn't expose .lower)
    graphdef, state = nnx.split(ts)

    @jax.jit
    def step_fn(state):
        ts = nnx.merge(graphdef, state)
        ts, returns = train_block(ts)
        _, new_state = nnx.split(ts)
        return new_state, returns

    t0 = time.time()
    lowered = step_fn.lower(state)
    lower_time = time.time() - t0

    t0 = time.time()
    compiled = lowered.compile()
    compile_time = time.time() - t0
    logger.log_once({"time/lower": lower_time, "time/compile": compile_time})

    # Run training blocks
    logger.start_time = time.time()
    for block_idx in range(1, num_blocks + 1):
        state, _ = compiled(state)
        jax.block_until_ready(state)

        elapsed = time.time() - logger.start_time
        print(
            f"Block {block_idx}/{num_blocks} done "
            f"(update {block_idx * block_size}/{num_updates}, {elapsed:.1f}s)"
        )

        # Save checkpoint (state is already an nnx.State from the split/merge loop)
        if args.num_checkpoints > 0:
            import os

            global_step = block_idx * block_size * steps_per_update
            cpu = jax.devices("cpu")[0]
            ckpt_state = jax.tree.map(lambda x: jax.device_put(x, cpu), state)
            path = os.path.join(logger.run_dir, "checkpoints", f"step_{global_step}")
            logger._checkpointer.save(path, ckpt_state)
            print(f"Started saving checkpoint: {path}")

    total_time = time.time() - logger.start_time
    logger.log_once({"time/total": total_time})
    print(f"Total training time: {total_time:.2f}s")
    logger.finish()

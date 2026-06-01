import dataclasses
import time

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import envelope
from envelope.typing import PyTree
from ppo_sharding.discretize_action_wrapper import DiscretizeActionWrapper
from ppo_sharding.logger import Logger
from ppo_sharding.networks import (
    DiscretePolicy,
    GaussianPolicy,
    ValueFunction,
    symexp,
    symlog,
)


DEFAULT_MAX_STEPS = 1000


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    total_timesteps: int = 1_000_000

    policy_lr: float = 0.0003
    policy_wd: float = 0.0001
    value_fn_lr: float = 0.0001
    value_wd: float = 0.0001

    entropy_coef: float = 0.01
    num_envs: int = 2000
    pool_size: int = 20
    num_steps: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Value function can optionally take more gradient steps per batch
    value_epochs: int = 1

    normalize_observations: bool = True
    normalize_rewards: bool = False
    discretize_actions: bool = False
    adam_epsilon: float = 1e-5
    seed: int = 0
    stagger_steps: int = 1

    # network arch
    activation: str = "tanh"
    layer_size: int = 256
    use_symlog: bool = False
    optimizer: str = "adamw"
    num_layers: int = 3
    layer_norm: bool = True
    init: str = "orthogonal"
    initial_log_std: float = 0.0

    # logging
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-reinforce"
    log_every: int = 131_072

    num_runs: int = 1

    # gradient clipping
    max_grad_norm: float = 0.5

    # learning rate schedule
    anneal_lr: bool = False

    # checkpointing
    num_checkpoints: int = 0
    log_dir: str = "runs"


def make_env(args: Args):
    env_kwargs = {}
    if args.env_name.startswith("kinetix::"):
        from kinetix.environment import ActionType
        env_kwargs.update(action_type=ActionType.MULTI_DISCRETE)
        print("Creating kinetix env with multi-discrete action space")
    if args.env_name.startswith("brax::"):
        env_kwargs.update(backend="mjx")
        print("Creating brax env with mjx backend")

    env = envelope.create(args.env_name, env_kwargs=env_kwargs)
    if not hasattr(env, "max_steps"):
        env = envelope.TruncationWrapper(env, max_steps=DEFAULT_MAX_STEPS)
    env = envelope.ContinuousObservationWrapper(env)
    env = envelope.FlattenObservationWrapper(env)
    env = envelope.FlattenActionWrapper(env)
    env = envelope.ClipActionWrapper(env)
    env = envelope.EpisodeStatisticsWrapper(env)
    if args.discretize_actions:
        env = DiscretizeActionWrapper(env)

    vecenv = envelope.PooledInitVmapWrapper(
        env,
        batch_size=args.num_envs,
        pool_size=args.pool_size,
    )
    if args.normalize_observations:
        vecenv = envelope.ObservationNormalizationWrapper(vecenv)
    if args.normalize_rewards:
        vecenv = envelope.RewardNormalizationWrapper(vecenv, discount=args.gamma)
    return env, vecenv


class TrainState(nnx.Pytree):
    def __init__(self, args: Args, seed: int | None = None, env_vecenv=None):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

        seed = seed if seed is not None else args.seed

        if env_vecenv is not None:
            env, vecenv = env_vecenv
        else:
            env, vecenv = make_env(args)

        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(seed)

        discrete = isinstance(env.action_space, envelope.Discrete)
        policy_cls = DiscretePolicy if discrete else GaussianPolicy
        policy_kwargs = dict(
            activation=args.activation,
            layer_size=args.layer_size,
            num_layers=args.num_layers,
            layer_norm=args.layer_norm,
            init=args.init,
        )
        if not discrete:
            policy_kwargs["initial_log_std"] = args.initial_log_std

        self.policy = policy_cls(
            env.observation_space,
            env.action_space,
            self.rngs,
            **policy_kwargs,
        )
        self.value_fn = ValueFunction(
            env.observation_space,
            self.rngs,
            activation=args.activation,
            layer_size=args.layer_size,
            use_symexp=args.use_symlog,
            num_layers=args.num_layers,
            layer_norm=args.layer_norm,
            init=args.init,
        )

        # Optimizer setup
        num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
        # Policy: 1 gradient step per update. Value: value_epochs steps per update.
        total_policy_steps = num_updates
        total_value_steps = num_updates * args.value_epochs

        if args.anneal_lr:
            policy_lr = optax.linear_schedule(args.policy_lr, 0.0, total_policy_steps)
            value_lr = optax.linear_schedule(args.value_fn_lr, 0.0, total_value_steps)
        else:
            policy_lr = args.policy_lr
            value_lr = args.value_fn_lr

        try:
            optimizer = getattr(optax, args.optimizer)
        except AttributeError:
            optimizer = getattr(optax.contrib, args.optimizer)

        policy_opt = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optimizer(policy_lr, eps=args.adam_epsilon, weight_decay=args.policy_wd),
        )
        self.policy_optimizer = nnx.Optimizer(self.policy, policy_opt, wrt=nnx.Param)

        value_opt = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optimizer(value_lr, eps=args.adam_epsilon, weight_decay=args.value_wd),
        )
        self.value_fn_optimizer = nnx.Optimizer(self.value_fn, value_opt, wrt=nnx.Param)

        # Initialize environment
        env_state, env_info = self.vecenv.init(self.rngs())
        if args.stagger_steps > 0:
            num_stagger_steps = args.stagger_steps * jnp.arange(args.num_envs)
            num_stagger_steps = num_stagger_steps % self.vecenv.max_steps
            env_state = stagger_env_state(env_state, num_stagger_steps)
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


def stagger_env_state(env_state, num_steps):
    if isinstance(env_state, envelope.TruncationWrapper.TruncationState):
        return env_state.replace(steps=num_steps)
    if isinstance(env_state, envelope.WrappedState):
        inner_state = stagger_env_state(env_state.inner_state, num_steps)
        return env_state.replace(inner_state=inner_state)
    raise ValueError("No TruncationState found in wrapper stack.")


def make_train_state(args: Args, seed: int | None = None):
    t0 = time.time()
    env_vecenv = make_env(args)
    make_env_time = time.time() - t0
    ts = TrainState(args, seed=seed, env_vecenv=env_vecenv)
    return ts, make_env_time


def apply_state_sharding(state, mesh, num_envs):
    env_sharding = NamedSharding(mesh, P("envs"))
    replicated = NamedSharding(mesh, P())

    def shard_leaf(path, x):
        path_str = jax.tree_util.keystr(path)
        is_env = "env_state" in path_str or "env_info" in path_str
        if is_env and x.ndim >= 1 and x.shape[0] == num_envs:
            return jax.device_put(x, env_sharding)
        return jax.device_put(x, replicated)

    return jax.tree_util.tree_map_with_path(shard_leaf, state)


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
        ts.env_info = env_info
        out_info = env_info.update(
            obs=obs,
            value=value,
            action=action,
            log_prob=pi.log_prob(action),
            value_next=ts.value_fn(env_info.final.obs),
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

    done = ts.env_info.terminated | ts.env_info.truncated
    last_obs = jnp.where(done[:, None], ts.env_info.final.obs, ts.env_info.obs)
    last_value = ts.value_fn(last_obs)
    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = gae_step(init_carry, info)
    return advantages


def flatten_batch(data: PyTree):
    """Flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)"""
    return jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), data)


def update_policy(ts: TrainState, batch):
    """Single full-batch REINFORCE gradient step.

    Since theta == theta_old (no previous updates within this batch),
    the importance sampling ratio is identically 1, and the surrogate
    objective reduces to: log pi_theta(a|s) * A_hat

    This is vanilla REINFORCE with GAE baseline.
    """

    @nnx.value_and_grad(has_aux=True)
    def loss_fn(policy):
        pi = policy(batch.obs)
        log_prob = pi.log_prob(batch.action)
        entropy = pi.entropy().mean()

        # Pure policy gradient: -E[log pi(a|s) * A]
        policy_loss = -jnp.mean(log_prob * batch.advantages)

        loss = policy_loss - ts.args.entropy_coef * entropy

        metrics = {
            "policy/loss": policy_loss,
            "policy/entropy": entropy,
        }
        return loss, metrics

    (loss, loss_metrics), grads = loss_fn(ts.policy)
    ts.policy_optimizer.update(ts.policy, grads)

    _, state = nnx.split(ts.policy)
    param_norm = optax.global_norm(state)
    grad_norm = optax.global_norm(grads)

    return {
        "policy/total_loss": loss,
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
    ts.value_fn_optimizer.update(ts.value_fn, grads)

    grad_norm = optax.global_norm(grads)
    _, state = nnx.split(ts.value_fn)
    param_norm = optax.global_norm(state)

    return {
        "value/loss": loss,
        "value/grad_norm": grad_norm,
        "value/param_norm": param_norm,
        "value/mean_prediction": values,
    }


def train_step(ts: TrainState):
    # Collect trajectories
    info = collect_trajectories(ts)

    # Compute advantages
    advantages = calculate_gae(ts, info)

    # Normalize advantages over the full batch (not per-minibatch)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    info = info.update(advantages=advantages)

    # Flatten to single batch: (num_steps, num_envs, ...) -> (batch_size, ...)
    batch = flatten_batch(info)

    # Single policy gradient step over the full batch
    policy_metrics = update_policy(ts, batch)

    # Value function update(s) — optionally multiple passes
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.value_epochs)
    def value_update_scan(ts):
        value_metrics = update_value_fn(ts, batch)
        return ts, value_metrics

    ts, value_metrics_all = value_update_scan(ts)
    # Average metrics across value epochs
    value_metrics = jax.tree.map(jnp.mean, value_metrics_all)

    # Explained variance diagnostic
    targets = batch.value + batch.advantages
    ev = 1.0 - jnp.var(targets - batch.value) / (jnp.var(targets) + 1e-8)

    loss_infos = {
        **policy_metrics,
        **value_metrics,
        "diagnostics/explained_variance": ev,
    }

    return info.update(loss_infos=loss_infos)


def make_block_fn(block_size: int, logger: Logger):
    @nnx.scan(in_axes=nnx.Carry, length=block_size)
    def train_block(ts: TrainState):
        out_info = train_step(ts)
        mean_return = jnp.nanmean(out_info.final.stats.reward)
        std_return = jnp.nanstd(out_info.final.stats.reward)
        mean_episode_length = jnp.nanmean(out_info.final.stats.length)
        std_episode_length = jnp.nanstd(out_info.final.stats.length)

        metrics = {
            "episode/return": mean_return,
            "episode/return_std": std_return,
            "episode/length": mean_episode_length,
            "episode/length_std": std_episode_length,
            **out_info.loss_infos,
        }
        jax.debug.callback(logger.log, ts.global_steps, metrics)
        return ts, mean_return

    return train_block


def make_sharded_step(graphdef, state, train_block, args):
    devices = jax.devices()
    num_devices = len(devices)
    assert args.num_envs % num_devices == 0, (
        f"num_envs ({args.num_envs}) must be divisible by num_devices ({num_devices})"
    )
    mesh = Mesh(devices, axis_names=("envs",))

    state = apply_state_sharding(state, mesh, args.num_envs)

    @jax.jit
    def step(state):
        ts = nnx.merge(graphdef, state)
        ts, returns = train_block(ts)
        _, new_state = nnx.split(ts)
        return new_state, returns

    t0 = time.time()
    lowered = step.lower(state)
    lower_time = time.time() - t0
    t0 = time.time()
    compiled = lowered.compile()
    compile_time = time.time() - t0

    return compiled, state, lower_time, compile_time


if __name__ == "__main__":
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))

    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update
    num_blocks = max(args.num_checkpoints, 1)
    block_size = num_updates // num_blocks
    assert block_size * num_blocks == num_updates, (
        f"num_updates ({num_updates}) must be divisible by num_blocks ({num_blocks})"
    )

    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        if args.num_runs > 1:
            print(f"\n{'=' * 60}")
            print(f"Run {run_idx + 1}/{args.num_runs} (seed={seed})")
            print(f"{'=' * 60}\n")

        logger = Logger(args)

        ts, make_env_time = make_train_state(args, seed=seed)
        graphdef, state = nnx.split(ts)

        train_block = make_block_fn(block_size, logger)
        step_fn, state, lower_time, compile_time = make_sharded_step(
            graphdef, state, train_block, args
        )

        logger.log_once({
            "time/lower": lower_time,
            "time/compile": compile_time,
            "time/make_env": make_env_time,
        })

        logger.start_time = time.time()
        for block_idx in range(1, num_blocks + 1):
            state, _ = step_fn(state)
            jax.block_until_ready(state)
            elapsed = time.time() - logger.start_time
            print(
                f"Block {block_idx}/{num_blocks} done "
                f"(update {block_idx * block_size}/{num_updates}, {elapsed:.1f}s)"
            )

            if args.num_checkpoints > 0:
                global_step = block_idx * block_size * steps_per_update
                ts = nnx.merge(graphdef, state)
                logger.save_checkpoint(global_step, ts)

        total_time = time.time() - logger.start_time
        logger.log_once({"time/total": total_time})
        print(f"Total training time: {total_time:.2f}s")
        logger.finish()
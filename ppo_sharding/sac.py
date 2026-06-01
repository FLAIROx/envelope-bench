import dataclasses
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx
from jax.sharding import Mesh

import envelope
from ppo_sharding.logger import Logger
from ppo_sharding.networks import SquashedGaussianPolicy, Temperature, TwinQFunction
from ppo_sharding.ppo import (
    DEFAULT_MAX_STEPS,
    apply_state_sharding,
    print_sharding_info,
    stagger_env_state,
)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array


class ReplayBuffer(NamedTuple):
    data: Transition
    index: jax.Array
    full: jax.Array


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "mujoco_playground::PandaOpenCabinet"
    total_timesteps: int = 1_000_000
    num_envs: int = 8192
    pool_size: int = 20
    buffer_size: int = 1_000_000
    batch_size: int = 256
    updates_per_step: int = 1
    random_steps: int = 10_000
    gamma: float = 0.99
    polyak: float = 0.995
    target_entropy: float | None = None
    seed: int = 0
    stagger_steps: int = 1
    normalize_observations: bool = False

    # network arch
    activation: str = "relu"
    layer_size: int = 256
    num_layers: int = 2
    layer_norm: bool = False
    critic_layer_norm: bool = False
    init: str = "orthogonal"

    # optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    actor_wd: float = 0.0
    critic_wd: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = float("inf")

    # logging
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-sac"
    log_every: int = 20_000
    num_runs: int = 1
    num_checkpoints: int = 0
    log_dir: str = "runs"
    run_name: str | None = None


def make_env(args: Args):
    env_kwargs = {}
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

    vecenv = envelope.PooledInitVmapWrapper(
        env,
        batch_size=args.num_envs,
        pool_size=args.pool_size,
    )
    if args.normalize_observations:
        vecenv = envelope.ObservationNormalizationWrapper(vecenv)
    return env, vecenv


def make_replay_buffer(size: int, obs_space: envelope.Space, action_space: envelope.Space):
    data = Transition(
        obs=jnp.empty((size, *obs_space.shape), dtype=obs_space.dtype),
        action=jnp.empty((size, *action_space.shape), dtype=action_space.dtype),
        reward=jnp.empty((size,), dtype=jnp.float32),
        done=jnp.empty((size,), dtype=jnp.bool_),
        next_obs=jnp.empty((size, *obs_space.shape), dtype=obs_space.dtype),
    )
    return ReplayBuffer(data=data, index=jnp.array(0), full=jnp.array(False))


def replay_buffer_extend(buffer: ReplayBuffer, batch: Transition) -> ReplayBuffer:
    size = buffer.data.obs.shape[0]
    batch_size = batch.obs.shape[0]
    idx = (buffer.index + jnp.arange(batch_size)) % size
    data = jax.tree.map(lambda arr, b: arr.at[idx].set(b), buffer.data, batch)
    next_index = (buffer.index + batch_size) % size
    full = buffer.full | (buffer.index + batch_size >= size)
    return ReplayBuffer(data=data, index=next_index, full=full)


def replay_buffer_sample(buffer: ReplayBuffer, batch_size: int, key: jax.Array) -> Transition:
    size = buffer.data.obs.shape[0]
    num_entries = jnp.where(buffer.full, size, buffer.index)
    idx = jax.random.randint(key, (batch_size,), 0, num_entries)
    return jax.tree.map(lambda arr: arr[idx], buffer.data)


def make_optimizer(learning_rate: float, weight_decay: float, args: Args):
    return optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(learning_rate, eps=args.adam_epsilon, weight_decay=weight_decay),
    )


class TrainState(nnx.Pytree):
    def __init__(self, args: Args, seed: int | None = None, env_vecenv=None):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

        seed = seed if seed is not None else args.seed
        if env_vecenv is not None:
            env, vecenv = env_vecenv
        else:
            env, vecenv = make_env(args)
        if isinstance(env.action_space, envelope.Discrete):
            raise ValueError("ppo_sharding.sac currently supports continuous actions only.")

        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(seed)

        network_kwargs = dict(
            layer_size=args.layer_size,
            activation=args.activation,
            num_layers=args.num_layers,
            init=args.init,
        )
        self.actor = SquashedGaussianPolicy(
            env.observation_space,
            env.action_space,
            self.rngs,
            layer_norm=args.layer_norm,
            **network_kwargs,
        )
        self.critic = TwinQFunction(
            env.observation_space,
            env.action_space,
            self.rngs,
            layer_norm=args.critic_layer_norm,
            **network_kwargs,
        )
        self.target_critic = TwinQFunction(
            env.observation_space,
            env.action_space,
            self.rngs,
            layer_norm=args.critic_layer_norm,
            **network_kwargs,
        )
        copy_params(self.critic, self.target_critic)
        self.temperature = Temperature()

        self.actor_optimizer = nnx.Optimizer(
            self.actor,
            make_optimizer(args.actor_lr, args.actor_wd, args),
            wrt=nnx.Param,
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            make_optimizer(args.critic_lr, args.critic_wd, args),
            wrt=nnx.Param,
        )
        self.alpha_optimizer = nnx.Optimizer(
            self.temperature,
            optax.adam(args.alpha_lr, eps=args.adam_epsilon),
            wrt=nnx.Param,
        )

        env_state, env_info = self.vecenv.init(self.rngs())
        if args.stagger_steps > 0:
            num_stagger_steps = args.stagger_steps * jnp.arange(args.num_envs)
            num_stagger_steps = num_stagger_steps % self.vecenv.max_steps
            env_state = stagger_env_state(env_state, num_stagger_steps)

        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)
        self.replay_buffer = nnx.data(
            make_replay_buffer(args.buffer_size, env.observation_space, env.action_space)
        )


def copy_params(source: nnx.Module, target: nnx.Module):
    _, source_state = nnx.split(source)
    nnx.update(target, source_state)


def polyak_update(source: nnx.Module, target: nnx.Module, polyak: float):
    _, source_state = nnx.split(source)
    _, target_state = nnx.split(target)
    new_target_state = jax.tree.map(
        lambda src, tgt: polyak * tgt + (1.0 - polyak) * src,
        source_state,
        target_state,
    )
    nnx.update(target, new_target_state)


def make_train_state(args: Args, seed: int | None = None):
    t0 = time.time()
    env_vecenv = make_env(args)
    make_env_time = time.time() - t0
    ts = TrainState(args, seed=seed, env_vecenv=env_vecenv)
    return ts, make_env_time


def _where_mask(mask: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    return jnp.where(mask, x, y)


def collect_transition(ts: TrainState):
    obs = ts.env_info.obs
    policy_action, _ = ts.actor.sample_and_log_prob(obs, ts.rngs())
    random_action = ts.vecenv.action_space.sample(ts.rngs())
    use_random = ts.global_steps < ts.args.random_steps
    action = jnp.where(use_random, random_action, policy_action)

    env_state, env_info = ts.vecenv.step(ts.env_state, action)
    terminal = env_info.terminated
    next_obs = _where_mask(env_info.truncated, env_info.final.obs, env_info.obs)
    transition = Transition(
        obs=obs,
        action=action,
        reward=env_info.reward,
        done=terminal,
        next_obs=next_obs,
    )

    ts.replay_buffer = replay_buffer_extend(ts.replay_buffer, transition)
    ts.global_steps += ts.args.num_envs
    ts.env_state = env_state
    ts.env_info = env_info
    return env_info


def target_entropy(ts: TrainState, batch: Transition) -> jax.Array:
    if ts.args.target_entropy is not None:
        return jnp.asarray(ts.args.target_entropy, dtype=jnp.float32)
    return -jnp.asarray(batch.action.shape[-1], dtype=jnp.float32)


def update_critic(ts: TrainState, batch: Transition):
    alpha = jax.lax.stop_gradient(ts.temperature())
    next_action, next_log_prob = ts.actor.sample_and_log_prob(batch.next_obs, ts.rngs())
    target_q1, target_q2 = ts.target_critic(batch.next_obs, next_action)
    target_q = jnp.minimum(target_q1, target_q2) - alpha * next_log_prob
    target = batch.reward + ts.args.gamma * (1.0 - batch.done.astype(jnp.float32)) * target_q
    target = jax.lax.stop_gradient(target)

    @nnx.value_and_grad(has_aux=True)
    def loss_fn(critic):
        q1, q2 = critic(batch.obs, batch.action)
        loss = optax.l2_loss(q1, target).mean() + optax.l2_loss(q2, target).mean()
        return loss, (q1.mean(), q2.mean())

    (loss, (q1_mean, q2_mean)), grads = loss_fn(ts.critic)
    ts.critic_optimizer.update(ts.critic, grads)
    grad_norm = optax.global_norm(grads)
    _, state = nnx.split(ts.critic)
    param_norm = optax.global_norm(state)
    return {
        "critic/loss": loss,
        "value/loss": loss,
        "critic/q1": q1_mean,
        "critic/q2": q2_mean,
        "critic/grad_norm": grad_norm,
        "critic/param_norm": param_norm,
    }


def update_actor_and_alpha(ts: TrainState, batch: Transition):
    alpha = jax.lax.stop_gradient(ts.temperature())
    entropy_target = target_entropy(ts, batch)
    action_key = ts.rngs()

    @nnx.value_and_grad(has_aux=True)
    def actor_loss_fn(actor):
        action, log_prob = actor.sample_and_log_prob(batch.obs, action_key)
        q1, q2 = ts.critic(batch.obs, action)
        loss = (alpha * log_prob - jnp.minimum(q1, q2)).mean()
        return loss, log_prob

    (actor_loss, log_prob), actor_grads = actor_loss_fn(ts.actor)
    ts.actor_optimizer.update(ts.actor, actor_grads)

    log_prob = jax.lax.stop_gradient(log_prob)

    @nnx.value_and_grad(has_aux=True)
    def alpha_loss_fn(temperature):
        alpha_value = temperature()
        loss = -(alpha_value * (log_prob + entropy_target)).mean()
        return loss, alpha_value

    (alpha_loss, alpha_value), alpha_grads = alpha_loss_fn(ts.temperature)
    ts.alpha_optimizer.update(ts.temperature, alpha_grads)

    return {
        "actor/loss": actor_loss,
        "policy/loss": actor_loss,
        "actor/entropy": -log_prob.mean(),
        "actor/grad_norm": optax.global_norm(actor_grads),
        "alpha/loss": alpha_loss,
        "alpha/value": alpha_value,
        "alpha/grad_norm": optax.global_norm(alpha_grads),
    }


def update_sac(ts: TrainState):
    batch = replay_buffer_sample(ts.replay_buffer, ts.args.batch_size, ts.rngs())
    critic_info = update_critic(ts, batch)
    actor_info = update_actor_and_alpha(ts, batch)
    polyak_update(ts.critic, ts.target_critic, ts.args.polyak)
    return {**critic_info, **actor_info}


def train_step(ts: TrainState):
    env_info = collect_transition(ts)

    @nnx.scan(in_axes=nnx.Carry, length=ts.args.updates_per_step)
    def update_scan(ts: TrainState):
        loss_info = update_sac(ts)
        return ts, loss_info

    ts, loss_infos = update_scan(ts)
    loss_infos = jax.tree.map(jnp.mean, loss_infos)
    return env_info.update(loss_infos=loss_infos)


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
    print_sharding_info(state, mesh)

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
    assert args.buffer_size >= args.num_envs, (
        f"buffer_size ({args.buffer_size}) must be at least num_envs ({args.num_envs})"
    )

    num_iterations = args.total_timesteps // args.num_envs
    num_blocks = max(args.num_checkpoints, 1)
    block_size = num_iterations // num_blocks
    assert block_size * num_blocks == num_iterations, (
        f"num_iterations ({num_iterations}) must be divisible by num_blocks ({num_blocks})"
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
                f"(iteration {block_idx * block_size}/{num_iterations}, {elapsed:.1f}s)"
            )

            if args.num_checkpoints > 0:
                global_step = block_idx * block_size * args.num_envs
                ts = nnx.merge(graphdef, state)
                logger.save_checkpoint(global_step, ts)

        total_time = time.time() - logger.start_time
        logger.log_once({"time/total": total_time})
        print(f"Total training time: {total_time:.2f}s")
        logger.finish()

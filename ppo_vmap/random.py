import dataclasses
import time

import jax
import jax.numpy as jnp
import tyro
from flax import nnx

import envelope
from ppo_vmap.logger import Logger

DEFAULT_MAX_STEPS = 1000


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    total_timesteps: int = 400_000
    num_envs: int = 100
    pool_size: int = 20
    num_steps: int = DEFAULT_MAX_STEPS
    gamma: float = 0.99
    normalize_observations: bool = True
    normalize_rewards: bool = False
    seed: int = 0

    stagger_steps: int = 0

    # logging
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-ppo"
    log_every: int = 10_000

    # parallelism: pmap across devices, vmap within each device
    num_runs: int = 16


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
    def __init__(
        self, args: Args, seed: int | None = None, run_idx: int = 0, env_vecenv=None
    ):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)
        self.run_idx = jnp.array(run_idx)

        seed = seed if seed is not None else args.seed

        if env_vecenv is not None:
            env, vecenv = env_vecenv
        else:
            env, vecenv = make_env(args)
        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(seed)

        # Initialize environment state and info
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


def make_train_states(args: Args):
    """Create train states for parallel runs."""
    num_devices = jax.device_count()
    assert args.num_runs % num_devices == 0, (
        f"num_runs ({args.num_runs}) must be divisible by device_count ({num_devices})"
    )
    t0 = time.time()
    env_vecenv = make_env(args)
    make_env_time = time.time() - t0
    states = [
        TrainState(args, seed=args.seed + i, run_idx=i, env_vecenv=env_vecenv)
        for i in range(args.num_runs)
    ]

    runs_per_device = args.num_runs // num_devices
    splits = [nnx.split(s) for s in states]
    graphdef = splits[0][0]
    all_states = [s[1] for s in splits]
    batched_state = jax.tree.map(
        lambda *xs: jnp.stack(xs).reshape(num_devices, runs_per_device, *xs[0].shape),
        *all_states,
    )
    return graphdef, batched_state, make_env_time


def collect_trajectories(ts: TrainState):
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_steps)
    def step_env(ts: TrainState):
        action = ts.vecenv.action_space.sample(ts.rngs())
        env_state, env_info = ts.vecenv.step(ts.env_state, action)
        ts.global_steps += ts.args.num_envs
        ts.env_state = env_state
        ts.env_info = env_info
        return ts, env_info

    ts, out_info = step_env(ts)
    return out_info


def make_block_fn(block_size: int, logger: Logger):
    """Create a rollout block function that runs block_size collection steps."""

    @nnx.scan(in_axes=nnx.Carry, length=block_size)
    def rollout_block(ts: TrainState):
        out_info = collect_trajectories(ts)
        mean_return = out_info.final.stats.reward.mean()
        std_return = out_info.final.stats.reward.std()
        mean_episode_length = out_info.final.stats.length.mean()
        std_episode_length = out_info.final.stats.length.std()
        metrics = {
            "episode/return": mean_return,
            "episode/return_std": std_return,
            "episode/length": mean_episode_length,
            "episode/length_std": std_episode_length,
            "policy/loss": jnp.float32(0.0),
            "value/loss": jnp.float32(0.0),
        }
        jax.debug.callback(logger.log, ts.global_steps, ts.run_idx, metrics)
        return ts, mean_return

    return rollout_block


def make_pmap_vmap_step(graphdef, rollout_block, batched_state):
    vmapped_block = nnx.vmap(rollout_block)

    @jax.pmap
    def step(state):
        ts = nnx.merge(graphdef, state)
        ts, returns = vmapped_block(ts)
        _, new_state = nnx.split(ts)
        return new_state, returns

    t0 = time.time()
    lowered = step.lower(batched_state)
    lower_time = time.time() - t0

    t0 = time.time()
    compiled = lowered.compile()
    compile_time = time.time() - t0

    return compiled, lower_time, compile_time


if __name__ == "__main__":
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    logger = Logger(args)

    # Compute block structure
    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update
    num_blocks = 1
    block_size = num_updates

    # Create train states
    graphdef, batched_state, make_env_time = make_train_states(args)

    # Build pmap(vmap(scan)) step function
    rollout_block = make_block_fn(block_size, logger)
    step_fn, lower_time, compile_time = make_pmap_vmap_step(
        graphdef, rollout_block, batched_state
    )
    logger.log_once({"time/lower": lower_time, "time/compile": compile_time, "time/make_env": make_env_time})

    # Run rollout
    logger.start_time = time.time()
    batched_state, _ = step_fn(batched_state)
    jax.block_until_ready(batched_state)

    total_time = time.time() - logger.start_time
    logger.log_once({"time/total": total_time})
    print(f"Total rollout time: {total_time:.2f}s")
    logger.finish()

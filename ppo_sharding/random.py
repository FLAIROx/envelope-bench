import dataclasses
import time

import jax
import jax.numpy as jnp
import tyro
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import envelope
from ppo_sharding.logger import Logger
from ppo_sharding.ppo import apply_state_sharding, print_sharding_info

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

    # parallelism: shard num_envs across devices via NamedSharding + jit
    # num_runs is for sequential repetition with different seeds (not parallelized)
    num_runs: int = 1

    # logging
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
        self, args: Args, seed: int | None = None, env_vecenv=None
    ):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

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


def make_train_state(args: Args, seed: int | None = None):
    """Create a single TrainState for data-parallel random rollouts."""
    t0 = time.time()
    env_vecenv = make_env(args)
    make_env_time = time.time() - t0
    ts = TrainState(args, seed=seed, env_vecenv=env_vecenv)
    return ts, make_env_time


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
        jax.debug.callback(logger.log, ts.global_steps, metrics)
        return ts, mean_return

    return rollout_block


def make_sharded_step(graphdef, state, rollout_block, args):
    """Build a sharding+jit step function for data-parallel random rollouts."""
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
        ts, returns = rollout_block(ts)
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

    # Compute block structure
    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update
    num_blocks = 1
    block_size = num_updates

    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx

        if args.num_runs > 1:
            print(f"\n{'=' * 60}")
            print(f"Run {run_idx + 1}/{args.num_runs} (seed={seed})")
            print(f"{'=' * 60}\n")

        logger = Logger(args)

        # Create single train state
        ts, make_env_time = make_train_state(args, seed=seed)
        graphdef, state = nnx.split(ts)

        # Build sharded jit step function
        rollout_block = make_block_fn(block_size, logger)
        step_fn, state, lower_time, compile_time = make_sharded_step(
            graphdef, state, rollout_block, args
        )
        logger.log_once({
            "time/lower": lower_time,
            "time/compile": compile_time,
            "time/make_env": make_env_time,
        })

        # Run rollout
        logger.start_time = time.time()
        state, _ = step_fn(state)
        jax.block_until_ready(state)

        total_time = time.time() - logger.start_time
        logger.log_once({"time/total": total_time})
        print(f"Total rollout time: {total_time:.2f}s")
        logger.finish()

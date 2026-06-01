"""Gumbel AlphaZero training script for envelope-bench.

Entry point: python -m alpha_zero.az --env_name "gymnax::CartPole-v1"

Architecture mirrors ppo_sharding/ppo.py:
  - Data-parallel via jax.sharding (env batch sharded, params replicated)
  - nnx.scan-based training loop
  - Reuses ppo_sharding logger
"""

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
from envelope.wrappers.flatten_observation_wrapper import flatten_x
from alpha_zero.networks import AZNet
from alpha_zero.mcts import mcts_policy, _process_raw_obs
from ppo_sharding.logger import Logger
from ppo_sharding.ppo import apply_state_sharding, print_sharding_info

DEFAULT_MAX_STEPS = 1000


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    total_timesteps: int = 1_000_000

    # MCTS
    num_simulations: int = 32
    gumbel_scale: float = 1.0

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    num_envs: int = 256
    pool_size: int = 20
    num_steps: int = 20
    num_epochs: int = 4
    num_minibatches: int = 8
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Network
    activation: str = "swish"
    layer_size: int = 256
    num_layers: int = 3
    layer_norm: bool = True
    init: str = "orthogonal"

    # Observation normalization
    normalize_observations: bool = True

    # Logging
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-az"
    log_every: int = 131_072
    log_dir: str = "runs"

    # Parallelism & runs
    num_runs: int = 1
    seed: int = 0

    # Gradient clipping
    max_grad_norm: float = float("inf")

    # Checkpointing
    num_checkpoints: int = 0

    # Stagger
    stagger_steps: int = 1


def make_env(args: Args):
    """Create the training vecenv and the raw single-instance env for MCTS.

    Returns:
        (single_env, vecenv, raw_env_for_mcts)
        - single_env: base env with wrappers (for getting spaces)
        - vecenv: fully vectorized + autoreset env for trajectory collection
        - raw_env_for_mcts: minimal env for MCTS tree simulation
    """
    env = envelope.create(args.env_name)

    # Ensure discrete action space
    base_action_space = env.action_space
    if not isinstance(base_action_space, envelope.Discrete):
        raise ValueError(
            f"AlphaZero requires a Discrete action space, got {type(base_action_space)}. "
            f"Only discrete-action envs are supported."
        )

    # Store the bare unwrapped env for MCTS tree simulation.
    # This must match the state type returned by _get_unwrapped_state().
    # MCTS doesn't need truncation — it only simulates num_simulations steps.
    raw_env_for_mcts = env.unwrapped

    # Build the wrapper stack for training
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

    return env, vecenv, raw_env_for_mcts


def stagger_env_state(env_state, num_steps):
    """Stagger env states across the batch for better diversity."""
    if isinstance(env_state, envelope.TruncationWrapper.TruncationState):
        return env_state.replace(steps=num_steps)
    if isinstance(env_state, envelope.WrappedState):
        inner_state = stagger_env_state(env_state.inner_state, num_steps)
        return env_state.replace(inner_state=inner_state)
    raise ValueError("No TruncationState found in wrapper stack.")


def _get_obs_norm_params(vecenv, env_state):
    """Extract observation normalization parameters from the wrapper stack.

    The ObservationNormalizationWrapper stores a RunningMeanVar in
    env_state.rmv_state with .mean and .var (which may be pytrees).
    Since we apply this to already-flattened obs inside MCTS, we flatten
    the mean/var pytrees to match.

    Returns None if ObservationNormalizationWrapper is not present.
    """
    if isinstance(vecenv, envelope.ObservationNormalizationWrapper):
        rmv = env_state.rmv_state
        # rmv.mean and rmv.var may be pytrees — flatten them to 1D
        # to match the flattened observation vector
        mean_flat = flatten_x(rmv.mean)
        var_flat = flatten_x(rmv.var)
        return {"mean": mean_flat, "var": var_flat}
    return None


def _get_unwrapped_state(env_state):
    """Traverse the wrapper stack to get the innermost (raw env) state."""
    if hasattr(env_state, "unwrapped"):
        return env_state.unwrapped
    return env_state


class TrainState(nnx.Pytree):
    def __init__(
        self, args: Args, seed: int | None = None, env_vecenv_raw=None
    ):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

        seed = seed if seed is not None else args.seed

        # Initialize environments
        if env_vecenv_raw is not None:
            single_env, vecenv, raw_env = env_vecenv_raw
        else:
            single_env, vecenv, raw_env = make_env(args)

        self.vecenv = nnx.data(vecenv)
        self.raw_env = nnx.static(raw_env)  # static: env structure doesn't change
        self.rngs = nnx.Rngs(seed)

        # Determine number of actions from the base (unwrapped) action space
        action_space = single_env.action_space
        # After FlattenActionWrapper, action_space may be a flat Discrete
        # We need the total number of discrete actions
        num_actions = int(jnp.prod(jnp.asarray(action_space.n)))
        self.num_actions = nnx.static(num_actions)

        # Initialize network
        self.az_net = AZNet(
            obs_space=single_env.observation_space,
            num_actions=num_actions,
            rngs=self.rngs,
            layer_size=args.layer_size,
            num_layers=args.num_layers,
            activation=args.activation,
            layer_norm=args.layer_norm,
            init=args.init,
        )

        # Initialize optimizer
        num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
        total_opt_steps = num_updates * args.num_epochs * args.num_minibatches
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.learning_rate, eps=1e-5),
        )
        self.optimizer = nnx.Optimizer(self.az_net, optimizer, wrt=nnx.Param)

        # Initialize environment state
        env_state, env_info = self.vecenv.init(self.rngs())
        if args.stagger_steps > 0:
            num_stagger_steps = args.stagger_steps * jnp.arange(args.num_envs)
            num_stagger_steps = num_stagger_steps % self.vecenv.max_steps
            env_state = stagger_env_state(env_state, num_stagger_steps)

        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


def make_train_state(args: Args, seed: int | None = None):
    """Create a TrainState for data-parallel training."""
    t0 = time.time()
    env_vecenv_raw = make_env(args)
    make_env_time = time.time() - t0
    ts = TrainState(args, seed=seed, env_vecenv_raw=env_vecenv_raw)
    return ts, make_env_time


def shuffle_and_split(data: PyTree, num_minibatches: int, key: jax.Array):
    """Shuffle and split trajectory data into minibatches."""
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
    """Collect trajectories using MCTS for action selection.

    For each step:
      1. Run MCTS on each env to get improved policy (action_weights)
      2. Sample action from MCTS policy
      3. Step the vectorized env
      4. Store transition data
    """

    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_steps)
    def step_env(ts: TrainState):
        obs = ts.env_info.obs  # (num_envs, obs_dim) — already flat + normalized

        # Get value predictions for all envs
        _, value = ts.az_net(obs)  # (num_envs,)

        # Extract unwrapped env states for MCTS simulation
        unwrapped_states = _get_unwrapped_state(ts.env_state)

        # Extract obs norm params if applicable
        obs_norm_params = _get_obs_norm_params(ts.vecenv, ts.env_state)

        # Run MCTS for each env (vmapped)
        mcts_keys = jax.random.split(ts.rngs(), ts.args.num_envs)

        mcts_out = jax.vmap(
            lambda obs_i, state_i, key_i: mcts_policy(
                az_net=ts.az_net,
                raw_env=ts.raw_env,
                obs_flat=obs_i,
                env_state=state_i,
                rng_key=key_i,
                num_simulations=ts.args.num_simulations,
                discount=ts.args.gamma,
                gumbel_scale=ts.args.gumbel_scale,
                obs_norm_params=obs_norm_params,
            )
        )(obs, unwrapped_states, mcts_keys)

        action = mcts_out.action                    # (num_envs,)
        action_weights = mcts_out.action_weights    # (num_envs, num_actions)

        # Step the vectorized environment
        env_state, env_info = ts.vecenv.step(ts.env_state, action)
        ts.global_steps += ts.args.num_envs
        ts.env_state = env_state
        ts.env_info = env_info

        # Build output info for this step
        out_info = env_info.update(
            obs=obs,
            value=value,
            action_weights=jax.lax.stop_gradient(action_weights),
            value_next=ts.az_net(env_info.final.obs)[1],  # V(s') for bootstrapping
        )
        return ts, out_info

    ts, out_info = step_env(ts)
    return out_info


def compute_value_targets(ts: TrainState, info):
    """Compute n-step discounted return targets.

    Uses reverse scan, same approach as PPO GAE but with full returns
    (lambda=1, i.e., no GAE — just discounted returns).
    """

    @nnx.scan(reverse=True)
    def return_step(carry, transition):
        G = carry
        done = transition.terminated | transition.truncated

        # For truncated episodes, bootstrap from value of final obs
        next_val = jnp.where(transition.truncated, transition.value_next, G)
        next_val = jnp.where(transition.terminated, 0.0, next_val)

        G = transition.reward + ts.args.gamma * next_val
        return G, G

    # Bootstrap from the last observation
    done = ts.env_info.terminated | ts.env_info.truncated
    last_obs = jnp.where(done[:, None], ts.env_info.final.obs, ts.env_info.obs)
    _, last_value = ts.az_net(last_obs)
    _, value_targets = return_step(last_value, info)
    return value_targets


def update_az(ts: TrainState, batch):
    """Single minibatch update for AlphaZero."""

    @nnx.value_and_grad(has_aux=True)
    def loss_fn(az_net):
        logits, value = az_net(batch.obs)

        # Policy loss: cross-entropy with MCTS policy targets
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        policy_loss = -jnp.sum(batch.action_weights * log_probs, axis=-1)
        policy_loss = jnp.mean(policy_loss)

        # Value loss: MSE against discounted returns
        value_loss = jnp.mean((value - batch.value_targets) ** 2)

        total_loss = (
            ts.args.policy_loss_weight * policy_loss
            + ts.args.value_loss_weight * value_loss
        )

        metrics = {
            "az/policy_loss": policy_loss,
            "az/value_loss": value_loss,
            "az/total_loss": total_loss,
            "az/mean_value_pred": jnp.mean(value),
            "az/mean_value_target": jnp.mean(batch.value_targets),
        }
        return total_loss, metrics

    (loss, metrics), grads = loss_fn(ts.az_net)
    ts.optimizer.update(ts.az_net, grads)
    grad_norm = optax.global_norm(grads)
    _, state = nnx.split(ts.az_net)
    param_norm = optax.global_norm(state)
    return {
        "az/loss": loss,
        "az/grad_norm": grad_norm,
        "az/param_norm": param_norm,
        **metrics,
    }


def update_epoch(ts: TrainState, minibatches):
    """Run one epoch of minibatch updates."""

    @nnx.scan
    def update_minibatch(ts: TrainState, batch):
        loss_info = update_az(ts, batch)
        return ts, loss_info

    _, loss_info = update_minibatch(ts, minibatches)
    return loss_info


def train_step(ts: TrainState):
    """One full training step: collect → targets → update."""

    # Collect trajectories with MCTS
    info = collect_trajectories(ts)

    # Compute value targets
    value_targets = compute_value_targets(ts, info)
    info = info.update(value_targets=value_targets)

    # Multiple epochs of SGD updates
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_epochs)
    def update_epoch_scan(ts):
        minibatches = shuffle_and_split(info, ts.args.num_minibatches, ts.rngs())
        loss_info = update_epoch(ts, minibatches)
        return ts, loss_info

    ts, loss_infos = update_epoch_scan(ts)
    loss_infos = jax.tree.map(jnp.mean, loss_infos)
    return info.update(loss_infos=loss_infos)


def make_block_fn(block_size: int, logger: Logger):
    """Create a training block function that runs block_size AZ updates."""

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
            # Logger expects these keys for its print statement
            "policy/loss": out_info.loss_infos["az/policy_loss"],
            "value/loss": out_info.loss_infos["az/value_loss"],
            **out_info.loss_infos,
        }
        jax.debug.callback(logger.log, ts.global_steps, metrics)
        return ts, mean_return

    return train_block


def make_sharded_step(graphdef, state, train_block, args):
    """Build a sharding+jit step function for data-parallel AZ training."""
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

    # Compute block structure
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

        # Create train state
        ts, make_env_time = make_train_state(args, seed=seed)
        graphdef, state = nnx.split(ts)

        # Build sharded jit step function (AOT compiled)
        train_block = make_block_fn(block_size, logger)
        step_fn, state, lower_time, compile_time = make_sharded_step(
            graphdef, state, train_block, args
        )
        logger.log_once({
            "time/lower": lower_time,
            "time/compile": compile_time,
            "time/make_env": make_env_time,
        })

        # Run training blocks
        logger.start_time = time.time()
        for block_idx in range(1, num_blocks + 1):
            state, _ = step_fn(state)
            jax.block_until_ready(state)

            elapsed = time.time() - logger.start_time
            print(
                f"Block {block_idx}/{num_blocks} done "
                f"(update {block_idx * block_size}/{num_updates}, {elapsed:.1f}s)"
            )

            # Save checkpoint
            if args.num_checkpoints > 0:
                global_step = block_idx * block_size * steps_per_update
                ts = nnx.merge(graphdef, state)
                logger.save_checkpoint(global_step, ts)

        total_time = time.time() - logger.start_time
        logger.log_once({"time/total": total_time})
        print(f"Total training time: {total_time:.2f}s")
        logger.finish()

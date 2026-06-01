"""Generic MCTS wrapper around mctx.gumbel_muzero_policy.

Works with any discrete-action envelope environment. The recurrent_fn uses
the raw (unwrapped) envelope env for tree simulation, applying the same
observation processing (flatten + normalize) that the training pipeline uses.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import mctx

from envelope.environment import Environment
from envelope.wrappers.flatten_observation_wrapper import flatten_x
from alpha_zero.networks import AZNet


def _cast_obs_to_float(obs):
    """Cast integer observation arrays to float32, matching ContinuousObservationWrapper."""
    def _cast_leaf(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.float32)
        return x.astype(jnp.float32)
    return jax.tree.map(_cast_leaf, obs)


def _process_raw_obs(raw_obs, obs_norm_params=None):
    """Process a raw env observation to match what the training network sees.

    Steps (matching the wrapper chain):
      1. Cast ints → float (ContinuousObservationWrapper)
      2. Flatten to 1D vector (FlattenObservationWrapper)
      3. Normalize using running stats (ObservationNormalizationWrapper)

    Args:
        raw_obs: structured observation from raw_env.step()
        obs_norm_params: optional dict with 'mean' and 'var' arrays for normalization

    Returns:
        flat, normalized observation vector
    """
    obs = _cast_obs_to_float(raw_obs)
    obs = flatten_x(obs)
    if obs_norm_params is not None:
        mean = obs_norm_params["mean"]
        var = obs_norm_params["var"]
        std = jnp.sqrt(var)
        obs = (obs - mean) / (std + 1e-8)
    return obs


def mcts_policy(
    az_net: AZNet,
    raw_env: Environment,
    obs_flat: jax.Array,
    env_state: Any,
    rng_key: jax.Array,
    num_simulations: int,
    discount: float,
    gumbel_scale: float = 1.0,
    obs_norm_params: dict | None = None,
) -> mctx.PolicyOutput:
    """Run Gumbel MuZero MCTS and return the improved policy.

    This function is designed to be vmapped over the env batch dimension.
    Internally, it adds a batch dimension of 1 for mctx (which always expects
    batched inputs), then squeezes the output.

    Args:
        az_net: the AZNet network (shared policy + value heads)
        raw_env: unwrapped single-instance envelope env for tree simulation
        obs_flat: flat, normalized observation for the current state, shape (obs_dim,)
        env_state: unwrapped env state (single instance, no wrapper state)
        rng_key: PRNG key
        num_simulations: number of MCTS simulations
        discount: discount factor γ
        gumbel_scale: Gumbel noise scale (higher = more exploration)
        obs_norm_params: optional normalization stats {mean, var}

    Returns:
        mctx.PolicyOutput with .action (scalar) and .action_weights (num_actions,)
    """
    # Root evaluation — add batch dim for mctx
    obs_batched = obs_flat[None, ...]  # (1, obs_dim)
    root_logits, root_value = az_net(obs_batched)  # (1, num_actions), (1,)

    # Batch the env state: tree_map to add leading dim of 1
    env_state_batched = jax.tree.map(lambda x: x[None, ...], env_state)

    root = mctx.RootFnOutput(
        prior_logits=root_logits,   # (1, num_actions)
        value=root_value,           # (1,)
        embedding=env_state_batched,
    )

    def recurrent_fn(_params, _rng_key, action, embedding):
        """MCTS transition model using the real environment.

        mctx calls this with batched inputs: action (1,), embedding has leading dim 1.
        We vmap raw_env.step over the batch dim.
        """
        # embedding is batched: (1, ...), action is (1,)
        next_state, info = jax.vmap(raw_env.step)(embedding, action)

        # Process raw obs to match network input — vmap over batch
        next_obs_flat = jax.vmap(
            lambda obs: _process_raw_obs(obs, obs_norm_params)
        )(info.obs)  # (1, obs_dim)

        next_logits, next_value = az_net(next_obs_flat)  # (1, num_actions), (1,)

        reward = jnp.asarray(info.reward, dtype=jnp.float32)      # (1,)
        terminated = jnp.asarray(info.terminated, dtype=jnp.bool_)  # (1,)

        # Zero out value for terminal states
        next_value = jnp.where(terminated, 0.0, next_value)
        step_discount = discount * (1.0 - terminated.astype(jnp.float32))

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=step_discount,
            prior_logits=next_logits,
            value=next_value,
        )
        return recurrent_output, next_state

    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=None,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=gumbel_scale,
    )

    # Squeeze batch dim from outputs: (1,) -> scalar, (1, A) -> (A,)
    action = policy_output.action[0]
    action_weights = policy_output.action_weights[0]

    return policy_output.replace(action=action, action_weights=action_weights)


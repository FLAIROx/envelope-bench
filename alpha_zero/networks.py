"""AlphaZero dual-head network (policy + value) using Flax NNX.

Architecture: shared MLP trunk → policy logits head + scalar value head.
Reuses helpers from ppo_sharding/networks.py for consistency.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import envelope
from ppo_sharding.networks import build_mlp_layers, ortho_linear


class AZNet(nnx.Module):
    """AlphaZero-style network with shared trunk, policy head, and value head.

    Input:  flat observation vector (obs_dim,)
    Output: (logits (num_actions,), value scalar)
    """

    def __init__(
        self,
        obs_space: envelope.Space,
        num_actions: int,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        num_layers: int = 3,
        activation: str = "swish",
        layer_norm: bool = True,
        init: str = "orthogonal",
    ):
        in_dim = int(np.prod(obs_space.shape))
        self.num_actions = num_actions

        # Shared MLP trunk
        trunk = build_mlp_layers(
            in_dim, layer_size, num_layers, rngs, activation, layer_norm, init
        )
        self.trunk = nnx.Sequential(*trunk)

        # Policy head: trunk output → logits
        self.policy_head = ortho_linear(layer_size, num_actions, rngs, scale=0.01)

        # Value head: trunk output → scalar in [-1, 1]
        self.value_head = nnx.Sequential(
            ortho_linear(layer_size, layer_size, rngs, scale=jnp.sqrt(2)),
            nnx.relu,
            ortho_linear(layer_size, 1, rngs, scale=1.0),
        )

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            obs: flat observation, shape (..., obs_dim)

        Returns:
            logits: shape (..., num_actions)
            value: shape (...)
        """
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = jnp.tanh(self.value_head(features).squeeze(-1))
        return logits, value

from functools import cached_property

import jax
import jax.numpy as jnp
from typing_extensions import override

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, Space, peel_batched
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import Wrapper


def flatten_space(space: Space):
    def is_leaf(x):
        # Tuples containing only integers are shape tuples (leaves)
        # PyTreeSpace can only have tuples that contain at least a Space, so
        # tuples with only integers must be shape tuples from leaf spaces
        return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

    shapes, treedef = jax.tree.flatten(space.shape, is_leaf=is_leaf)
    dims = [jnp.prod(jnp.asarray(shape)) for shape in shapes]
    return treedef, shapes, dims


def flatten_x(x: PyTree, num_batch_dims: int = 0):
    """Flatten obs leaves and concatenate into a single vector.

    The first ``num_batch_dims`` axes are preserved; the remaining axes of each
    leaf are flattened, then concatenated along the last axis.
    """
    leaves = jax.tree.leaves(x)
    arrs = [jnp.asarray(leaf) for leaf in leaves]
    batch_shape = arrs[0].shape[:num_batch_dims]
    xs = [a.reshape(*batch_shape, -1) for a in arrs]
    return jnp.concatenate(xs, axis=-1)


class FlattenObservationWrapper(Wrapper):
    @cached_property
    def _num_batch_dims(self) -> int:
        batch_dims, _ = peel_batched(self.env.observation_space)
        return len(batch_dims)

    def _flatten_info_obs(self, info: Info) -> Info:
        """Flatten obs in info, and in info.final if present."""
        n = self._num_batch_dims
        info = info.update(obs=flatten_x(info.obs, n))
        if hasattr(info, "final"):
            flat_final = info.final.update(obs=flatten_x(info.final.obs, n))
            info = info.update(final=flat_final)
        return info

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.init(key)
        return state, self._flatten_info_obs(info)

    @override
    def reset(self, key: Key, state: State) -> tuple[State, Info]:
        state, info = self.env.reset(key, state)
        return state, self._flatten_info_obs(info)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        return state, self._flatten_info_obs(info)

    @override
    @cached_property
    def observation_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.observation_space)

        def is_leaf(x):
            spaces = (Continuous, Discrete)
            return isinstance(x, spaces)

        spaces = jax.tree.leaves(base, is_leaf=is_leaf)
        obs_cls = type(spaces[0])

        if not all(isinstance(space, obs_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

        if obs_cls == Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            space = Continuous(low=low, high=high)
        elif obs_cls == Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            space = Discrete(n=n)
        else:
            raise ValueError(f"Unsupported space type: {obs_cls}")

        for batch_dim in batch_dims:
            space = BatchedSpace(space, batch_dim)
        return space

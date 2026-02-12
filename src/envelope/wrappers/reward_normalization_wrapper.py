from typing import override
from functools import cached_property

import jax
from jax import numpy as jnp

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.normalization import update_rmv, RunningMeanVar
from envelope.wrappers.wrapper import WrappedState, Wrapper
from envelope.typing import Array
from envelope.struct import FrozenPyTreeNode, static_field


class RewardRunningMeanVar(RunningMeanVar):
    returns: PyTree


class RewardNormalizationWrapper(Wrapper):
    class RewardNormalizationState(WrappedState):
        rmv_state: RewardRunningMeanVar = field()

    discount: float = static_field(default=0.99)

    def _init_rmv_state(self) -> RewardRunningMeanVar:

        return RewardRunningMeanVar(
            mean=jnp.zeros(1),
            var=jnp.ones(1),
            count=jnp.asarray(0),
            returns=jnp.zeros(1),
        )

    def _normalize_rew(self, reward: PyTree, rmv: RewardRunningMeanVar) -> PyTree:
        def norm_leaf(x, mean, std):
            mean = jnp.broadcast_to(mean, x.shape)
            std = jnp.broadcast_to(std, x.shape)
            rew = x / (std + 1e-8)
            return rew

        return jax.tree.map(norm_leaf, reward, rmv.mean, rmv.std)

    def _normalize_and_update(
        self, state: WrappedState, info: Info
    ) -> tuple[WrappedState, Info]:
        returns = (
            info.reward
            + (1 - info.terminated) * self.discount * state.rmv_state.returns
        )

        rmv_state = update_rmv(state.rmv_state, returns)
        rmv_state = RewardRunningMeanVar(
            mean=rmv_state.mean,
            var=rmv_state.var,
            count=rmv_state.count,
            returns=returns,
        )
        norm_rew = self._normalize_rew(info.reward, rmv_state)

        state = self.RewardNormalizationState(
            inner_state=state.inner_state, rmv_state=rmv_state
        )
        info = info.update(reward=norm_rew, unnormalized_reward=info.reward)
        return state, info

    @override
    def init(self, key: Key) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.init(key)
        rmv_state = self._init_rmv_state()
        next_state = self.RewardNormalizationState(
            inner_state=inner_state, rmv_state=rmv_state
        )
        return self._normalize_and_update(next_state, info)

    @override
    def reset(self, state: WrappedState, key: Key) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        # Preserve running statistics across resets
        next_state = self.RewardNormalizationState(
            inner_state=inner_state, rmv_state=state.rmv_state
        )
        return self._normalize_and_update(next_state, info)

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        state = state.replace(inner_state=inner_state)
        return self._normalize_and_update(state, info)

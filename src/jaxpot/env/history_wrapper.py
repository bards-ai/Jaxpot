from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class HistoryWrappedState:
    """Vectorized env state with a rolling observation history."""

    env_state: Any
    observation: Any

    @property
    def legal_action_mask(self):
        return self.env_state.legal_action_mask

    @property
    def current_player(self):
        return self.env_state.current_player

    @property
    def rewards(self):
        return self.env_state.rewards

    @property
    def terminated(self):
        return self.env_state.terminated

    @property
    def truncated(self):
        return self.env_state.truncated


def _repeat_obs_history(obs: Any, history_len: int) -> Any:
    return jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], repeats=history_len, axis=1),
        obs,
    )


def _expand_done_mask(done: jnp.ndarray, ndim: int) -> jnp.ndarray:
    return done.reshape(done.shape + (1,) * (ndim - 1))


def wrap_vectorized_env_with_history(
    init_fn: Callable,
    step_fn: Callable,
    *,
    history_len: int,
) -> tuple[Callable, Callable]:
    """Wrap vectorized PGX init/step functions with rolling observation history.

    Parameters
    ----------
    init_fn
        Vectorized init function with signature ``init_fn(keys) -> state``.
    step_fn
        Vectorized step function with signature ``step_fn(state, actions, keys) -> state``.
    history_len
        Number of recent observations to keep in the returned state.
    """
    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")

    def init_wrapped(keys):
        env_state = init_fn(keys)
        obs_hist = _repeat_obs_history(env_state.observation, history_len)
        return HistoryWrappedState(env_state=env_state, observation=obs_hist)

    def step_wrapped(state: HistoryWrappedState, actions, keys):
        env_state_n = step_fn(state.env_state, actions, keys)
        done_n = jnp.logical_or(env_state_n.terminated, env_state_n.truncated)

        def _update_hist(hist_x, obs_x):
            shifted = jnp.concatenate([hist_x[:, 1:, ...], obs_x[:, None, ...]], axis=1)
            reset = jnp.repeat(obs_x[:, None, ...], repeats=history_len, axis=1)
            done_mask = _expand_done_mask(done_n, shifted.ndim)
            return jnp.where(done_mask, reset, shifted)

        obs_hist_n = jax.tree.map(_update_hist, state.observation, env_state_n.observation)
        return HistoryWrappedState(env_state=env_state_n, observation=obs_hist_n)

    return init_wrapped, step_wrapped

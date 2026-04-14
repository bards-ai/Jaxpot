"""Wrapper that augments PGX observations with action history.

PGX default observations for small poker games (Kuhn, Leduc) don't include
action history, causing observation aliasing — multiple distinct information
states map to the same observation. This wrapper appends one-hot encoded
action history to observations, eliminating aliasing.

Usage in config:
    env_action_history_len: 6  # max actions to track (Kuhn: 3, Leduc: 6)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ActionHistoryState:
    """Wraps PGX env state with action history tracking."""

    env_state: ...
    # action_history: [max_actions, num_actions+1] one-hot per step
    # Extra dim for "no action yet" sentinel
    action_history: jnp.ndarray
    # How many actions have been taken so far
    action_count: jnp.ndarray
    # Flattened observation: original obs + action history
    observation: jnp.ndarray

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


def wrap_with_action_history(
    env,
    init_fn: Callable,
    step_fn: Callable,
    *,
    max_actions: int,
) -> tuple[Callable, Callable, int]:
    """Wrap vectorized PGX init/step to append action history to observations.

    Parameters
    ----------
    env
        PGX environment (used to get num_actions).
    init_fn
        Vectorized init function: init_fn(keys) -> state.
    step_fn
        Vectorized step function: step_fn(state, actions, keys) -> state.
    max_actions
        Maximum number of actions to track in history.

    Returns
    -------
    init_wrapped, step_wrapped, new_obs_size
        Wrapped functions and the new observation dimension.
    """
    num_actions = env.num_actions
    # Each history slot: one-hot over (num_actions + 1) where last = "empty"
    history_slot_size = num_actions + 1
    base_obs_size = env.observation_shape[0] if hasattr(env, 'observation_shape') else None

    def _make_obs(env_obs, action_history):
        """Concatenate base observation with flattened action history."""
        # env_obs: [batch, obs_dim] (float32)
        # action_history: [batch, max_actions, num_actions+1]
        flat_hist = action_history.reshape(env_obs.shape[0], -1)
        return jnp.concatenate([env_obs.astype(jnp.float32), flat_hist], axis=-1)

    def init_wrapped(keys):
        env_state = init_fn(keys)
        batch_size = keys.shape[0]
        # Initialize history: all slots point to "empty" (last index)
        action_history = jnp.zeros(
            (batch_size, max_actions, history_slot_size), dtype=jnp.float32
        )
        # Set "empty" sentinel for all slots
        action_history = action_history.at[:, :, -1].set(1.0)
        action_count = jnp.zeros(batch_size, dtype=jnp.int32)
        obs = _make_obs(env_state.observation, action_history)
        return ActionHistoryState(
            env_state=env_state,
            action_history=action_history,
            action_count=action_count,
            observation=obs,
        )

    def step_wrapped(state: ActionHistoryState, actions, keys):
        env_state_n = step_fn(state.env_state, actions, keys)
        done_n = jnp.logical_or(env_state_n.terminated, env_state_n.truncated)
        batch_size = actions.shape[0]

        # Update action history: write one-hot of action at current position
        new_slot = jax.nn.one_hot(actions, history_slot_size)  # [batch, num_actions+1]
        # Clip action_count to max_actions-1 to avoid OOB
        write_idx = jnp.minimum(state.action_count, max_actions - 1)
        # Write into history at the current position
        new_history = state.action_history.at[
            jnp.arange(batch_size), write_idx
        ].set(new_slot)
        new_count = state.action_count + 1

        # On episode reset, clear history
        reset_history = jnp.zeros_like(new_history)
        reset_history = reset_history.at[:, :, -1].set(1.0)
        reset_count = jnp.zeros_like(new_count)

        # Select based on done
        done_3d = done_n[:, None, None]
        final_history = jnp.where(done_3d, reset_history, new_history)
        final_count = jnp.where(done_n, reset_count, new_count)

        obs = _make_obs(env_state_n.observation, final_history)
        return ActionHistoryState(
            env_state=env_state_n,
            action_history=final_history,
            action_count=final_count,
            observation=obs,
        )

    new_obs_size = max_actions * history_slot_size
    if base_obs_size is not None:
        new_obs_size += base_obs_size

    return init_wrapped, step_wrapped, new_obs_size

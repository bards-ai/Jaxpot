from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import struct

if TYPE_CHECKING:
    from jaxpot.rollout.aux_target_hooks import AuxTargetHook


@runtime_checkable
class RolloutAuxTransform(Protocol):
    """Derives one aux_target entry from RolloutBuffer fields at buffer-build time.

    Implement this to inject trainer-specific data (e.g. MCTS policy from
    policy_logits) into aux_targets during TrainingDataBuffer construction,
    without coupling buffer.py to any particular trainer.
    """

    target_field: str

    def __call__(self, rb: "RolloutBuffer", seats: tuple[int, ...]) -> jax.Array:
        """Return the flattened aux target array, shape [N, ...], N = total samples."""
        ...


class RolloutBuffer(struct.PyTreeNode):
    """Buffer for storing rollout trajectory data.

    The obs field can be either a PyTreeNode (structured observation) or
    a jax.Array (flat observation), depending on the observation type used.

    Shape conventions:
        ``[num_envs, num_steps, 2_players, ...]``
    Hidden state shape:
        ``[num_envs, num_steps, 2, num_layers, hidden_size]``
    """

    obs: jax.Array | struct.PyTreeNode
    value: jnp.ndarray
    policy_logits: jnp.ndarray
    legal_action_mask: jnp.ndarray
    done: jnp.ndarray
    terminated: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    log_prob: jnp.ndarray
    valids: jnp.ndarray
    aux_targets: dict[str, jnp.ndarray]
    hidden_state: jnp.ndarray

    def get_aux_target(self, target_name: str) -> jnp.ndarray:
        return self.aux_targets[target_name]

    def set_aux_target(self, target_name: str, value: jnp.ndarray) -> RolloutBuffer:
        updated = dict(self.aux_targets)
        updated[target_name] = value
        return self.replace(aux_targets=updated)


def init_aux_target_buffers(
    aux_target_hooks: tuple["AuxTargetHook", ...],
    num_envs: int,
    num_steps: int,
) -> dict[str, jnp.ndarray]:
    """Build fixed-key aux target buffers from hooks."""
    return {hook.target_field: hook.init_buffer(num_envs, num_steps) for hook in aux_target_hooks}


class TrainingDataBuffer(struct.PyTreeNode):
    """Sequence-based training data for PPO (both recurrent and non-recurrent).

    All trajectory fields have a leading ``[num_sequences, seq_len]`` shape.
    For non-recurrent models ``seq_len=1``. The ``hidden_state`` field carries
    the per-step recurrent state (opaque to this buffer); the first slice is
    used as the initial carry for recurrent training, and ``done`` flags
    allow mid-sequence hidden state resets.
    """

    obs: jax.Array | struct.PyTreeNode  # [N, seq_len, *obs_shape]
    value: jax.Array  # [N, seq_len, 1]
    returns: jax.Array  # [N, seq_len, 1]
    adv: jax.Array  # [N, seq_len]
    actions: jax.Array  # [N, seq_len]
    log_prob: jax.Array  # [N, seq_len]
    legal_action_mask: jax.Array  # [N, seq_len, A]
    valids: jax.Array  # [N, seq_len]
    done: jax.Array  # [N, seq_len]
    hidden_state: jax.Array  # [N, *hidden_state_shape] — initial carry per chunk
    value_loss_mask: jax.Array  # [N, seq_len]
    aux_targets: dict[str, jax.Array]

    def get_auxiliary_targets_for_sample(self, i: int) -> dict[str, jax.Array]:
        return {
            target_name: (
                self.aux_targets[target_name][i].tolist()
                if hasattr(self.aux_targets[target_name][i], "tolist")
                else int(self.aux_targets[target_name][i])
            )
            for target_name in self.aux_targets
        }


@partial(jax.jit, static_argnames=("seats", "seq_len", "rollout_transforms"))
def create_training_buffer(
    rb: RolloutBuffer,
    advantages: tuple[jax.Array, ...],
    seats: tuple[int, ...],
    seq_len: int = 1,
    rollout_transforms: tuple[RolloutAuxTransform, ...] = (),
    value_loss_masks: tuple[jax.Array, ...] = (),
) -> TrainingDataBuffer:
    """Create sequence-based training buffer from rollout data.

    Chunks per-player trajectories into ``seq_len``-length sequences.
    Requires ``num_steps`` to be divisible by ``seq_len``.

    For non-recurrent models, use ``seq_len=1`` (the default).
    """
    num_envs = rb.value.shape[0]
    num_steps = rb.value.shape[1]
    num_chunks = num_steps // seq_len

    def extract_seat_chunked(data: jax.Array, seat: int) -> jax.Array:
        """Extract seat data and reshape [E, T, ...] -> [E*C, S, ...]."""
        x = data[:, :, seat]  # [E, T, ...]
        extra = x.shape[2:]
        return x.reshape(num_envs, num_chunks, seq_len, *extra).reshape(
            num_envs * num_chunks, seq_len, *extra
        )

    def concat_seats_chunked(data: jax.Array) -> jax.Array:
        return jnp.concatenate([extract_seat_chunked(data, seat) for seat in seats], axis=0)

    obs = jax.tree.map(concat_seats_chunked, rb.obs)

    values_chunked = []
    returns_chunked = []
    for adv, seat in zip(advantages, seats, strict=True):
        val = rb.value[:, :, seat]  # [E, T]
        ret = adv + val  # [E, T]
        val_c = val.reshape(num_envs, num_chunks, seq_len).reshape(num_envs * num_chunks, seq_len)
        ret_c = ret.reshape(num_envs, num_chunks, seq_len).reshape(num_envs * num_chunks, seq_len)
        values_chunked.append(val_c)
        returns_chunked.append(ret_c)

    value = jnp.concatenate(values_chunked, axis=0)[..., None]  # [N, S, 1]
    returns = jnp.concatenate(returns_chunked, axis=0)[..., None]  # [N, S, 1]

    adv_chunked = []
    for adv_arr, seat in zip(advantages, seats, strict=True):
        a = adv_arr.reshape(num_envs, num_chunks, seq_len).reshape(num_envs * num_chunks, seq_len)
        adv_chunked.append(a)
    adv_all = jnp.concatenate(adv_chunked, axis=0)

    actions = concat_seats_chunked(rb.actions)
    log_prob = concat_seats_chunked(rb.log_prob)
    legal_action_mask = concat_seats_chunked(rb.legal_action_mask)
    valids = concat_seats_chunked(rb.valids).astype(jnp.float32)
    done = concat_seats_chunked(rb.done).astype(jnp.float32)

    # Extract initial hidden states: take the hidden at the start of each chunk
    def extract_init_hidden(h: jnp.ndarray) -> jnp.ndarray:
        """h shape: [E, T, 2, *state_shape] -> [N, *state_shape] via chunk starts."""
        parts = []
        for seat in seats:
            h_seat = h[:, :, seat]  # [E, T, *state_shape]
            # Take every seq_len-th step as chunk start
            h_starts = h_seat[:, ::seq_len]  # [E, C, *state_shape]
            parts.append(h_starts.reshape(num_envs * num_chunks, *h_starts.shape[2:]))
        return jnp.concatenate(parts, axis=0)

    hidden_state = extract_init_hidden(rb.hidden_state)

    # Aux targets: hook-collected fields are stored on the rollout buffer per
    # seat/timestep, then chunked alongside the trajectory. Rollout transforms
    # are applied at buffer-build time and contribute additional training
    # targets (e.g. MCTS policy targets for AlphaZero).
    aux_targets = {name: concat_seats_chunked(rb.aux_targets[name]) for name in rb.aux_targets}
    for transform in rollout_transforms:
        aux_targets[transform.target_field] = transform(rb, seats)

    return TrainingDataBuffer(
        obs=obs,
        value=value,
        returns=returns,
        adv=adv_all,
        actions=actions,
        log_prob=log_prob,
        legal_action_mask=legal_action_mask,
        valids=valids,
        done=done,
        hidden_state=hidden_state,
        value_loss_mask=jnp.ones_like(valids, dtype=jnp.float32),
        aux_targets=aux_targets,
    )


@jax.jit
def concatenate_training_data(
    batches: list[struct.PyTreeNode | None],
) -> struct.PyTreeNode:
    """Concatenate any PyTreeNode training buffers along axis 0.

    Works for TrainingDataBuffer or any other ``PyTreeNode`` with array leaves.
    """
    valid = [b for b in batches if b is not None]
    if not valid:
        raise ValueError("No valid batches to concatenate")
    if len(valid) == 1:
        return valid[0]
    return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *valid)

from functools import partial
from typing import Literal

import jax
from jax import numpy as jnp

from jaxpot.rollout.buffer import RolloutBuffer

ValueTargetMode = Literal["gae", "per_seat_return"]


@partial(jax.jit, static_argnames=["discount"])
def calculate_discounted_sum_jax(
    x: jax.Array,
    dones: jax.Array,
    discount: float,
    x_last: jax.Array | None = None,
    valids: jax.Array | None = None,
) -> jax.Array:
    """
    Discounted sum with episode termination.

    Parameters
    ----------
    x : jax.Array
        Sequence over time [T, ...].
    dones : jax.Array
        Done flags [T, ...].
    discount : float
        Discount factor.
    x_last : jax.Array | None
        Bootstrap for t = T.
    valids : jax.Array | None
        Optional step-valid mask [T, ...]. If provided, steps with valids=0
        do not apply discounting (carry is propagated) and x is assumed to be
        already zeroed for invalid steps.

    Returns
    -------
    jax.Array
        Discounted sums [T, ...].
    """
    x_dtype = x.dtype
    dones_f = dones.astype(x_dtype)
    valids_f = jnp.ones_like(x, dtype=x_dtype) if valids is None else valids.astype(x_dtype)
    init_carry = jnp.zeros_like(x[-1]) if x_last is None else x_last.astype(x_dtype)

    def body(carry, inputs):
        xi, di, vi = inputs
        discount_valid = discount * vi + (1.0 - vi)
        new_carry = xi + discount_valid * carry * (1.0 - di)
        return new_carry, new_carry

    inputs_rev = (
        jnp.flip(x, axis=0),
        jnp.flip(dones_f, axis=0),
        jnp.flip(valids_f, axis=0),
    )
    _, ys_rev = jax.lax.scan(body, init_carry, inputs_rev)
    return jnp.flip(ys_rev, axis=0)


@partial(jax.jit, static_argnames=["gamma", "lam"])
def gae_advantages(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    valids: jax.Array,
    gamma: float,
    lam: float,
) -> jax.Array:
    """
    Generalized Advantage Estimation for terminal-only trajectories.

    Assumes all valid samples end with done=True, so no bootstrap value is needed
    (the bootstrap term is zeroed by (1 - done) for terminal states).

    Parameters
    ----------
    rewards : jax.Array
        [E, T]
    dones : jax.Array
        [E, T]
    values : jax.Array
        [E, T]
    valids : jax.Array
        Step-valid mask [E, T]. Used to mask deltas and control discounting.
    gamma : float
        Discount.
    lam : float
        GAE lambda.

    Returns
    -------
    jax.Array
        Advantages [E, T].
    """
    rewards_t = jnp.swapaxes(rewards, 0, 1)
    dones_t = jnp.swapaxes(dones, 0, 1).astype(rewards_t.dtype)
    values_t = jnp.swapaxes(values, 0, 1)
    valids_t = jnp.swapaxes(valids, 0, 1).astype(rewards_t.dtype)

    # Shift values to get V(s') and append 0 for terminal states (where done=True)
    values_next = jnp.concatenate([values_t[1:], jnp.zeros_like(values_t[:1])], axis=0)
    deltas = rewards_t + (1.0 - dones_t) * gamma * values_next - values_t
    deltas = deltas * valids_t
    adv_t = calculate_discounted_sum_jax(deltas, dones_t, gamma * lam, valids=valids_t)
    return jnp.swapaxes(adv_t, 0, 1)


def gae_advantages_from_rollout_buffer(
    rollout_buffer: RolloutBuffer,
    seat: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> jax.Array:
    """Compute GAE advantages from rollout buffer."""
    return gae_advantages(
        rollout_buffer.rewards[:, :, seat],
        rollout_buffer.done[:, :, seat],
        rollout_buffer.value[:, :, seat],
        rollout_buffer.valids[:, :, seat],
        gamma=gamma,
        lam=gae_lambda,
    )


@partial(jax.jit, static_argnames=["num_steps", "backup_gamma"])
def per_seat_returns_and_value_mask(
    rewards: jax.Array,
    terminated: jax.Array,
    valids: jax.Array,
    num_steps: int,
    backup_gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """Value targets from a per-seat backward scan, equivalent to PGX AlphaZero targets.

    For each (env, per-seat-time): ``v_t = r_t + d_t * v_{t+1}`` with ``d_t = 0``
    when the game terminated, else ``d_t = +backup_gamma``.

    This gives identical value targets to the PGX reference's global-timeline
    alternating backup (``discount = -1`` per step).  The PGX sign-flip is not
    needed here because each seat processes only its own steps and sees its own
    signed terminal reward (the rollout collector back-fills opponent rewards).

    Truncated episodes (timeout without ``terminated``) are excluded from the value
    loss via the returned mask — matching the PGX ``cumsum(terminated) >= 1`` mask.

    Parameters
    ----------
    rewards, terminated, valids
        Per-seat slices ``[num_envs, num_steps]``.
    num_steps
        Horizon ``T`` (static for JIT).
    backup_gamma
        Discount applied between consecutive per-seat steps.  Use ``1.0`` for
        standard undiscounted games (Go, Connect4, etc.).

    Returns
    -------
    returns, value_loss_mask
        Both ``[num_envs, num_steps]``. Mask is 1 where the step is valid and a
        terminal outcome occurs at or after this timestep in the unroll (so
        truncated-incomplete trajectories are masked out).
    """
    r_dtype = rewards.dtype
    valid_f = valids.astype(r_dtype)
    term_b = terminated
    disc = jnp.where(term_b, 0.0, backup_gamma).astype(r_dtype)
    r_eff = rewards.astype(r_dtype) * valid_f
    disc_eff = jnp.where(valid_f > 0, disc, 0.0)

    r_T = jnp.swapaxes(r_eff, 0, 1)
    d_T = jnp.swapaxes(disc_eff, 0, 1)

    def body(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
        ix = num_steps - 1 - i
        v = r_T[ix] + d_T[ix] * carry
        return v, v

    init_carry = jnp.zeros(r_T.shape[1], dtype=r_dtype)
    _, vs = jax.lax.scan(body, init_carry, jnp.arange(num_steps, dtype=jnp.int32))
    vs_fwd = vs[::-1]
    returns = jnp.swapaxes(vs_fwd, 0, 1)

    term_f = term_b.astype(r_dtype)
    has_term_ahead = (jnp.cumsum(term_f[:, ::-1], axis=1)[:, ::-1] >= 1).astype(r_dtype)
    value_loss_mask = valid_f * has_term_ahead
    return returns, value_loss_mask


def per_seat_returns_from_rollout_buffer(
    rollout_buffer: RolloutBuffer,
    seat: int,
    num_steps: int,
    backup_gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """Per-seat terminal returns and value mask for one seat (PGX-equivalent)."""
    return per_seat_returns_and_value_mask(
        rollout_buffer.rewards[:, :, seat],
        rollout_buffer.terminated[:, :, seat],
        rollout_buffer.valids[:, :, seat],
        num_steps,
        backup_gamma,
    )

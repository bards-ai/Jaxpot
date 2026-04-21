from __future__ import annotations

import jax.numpy as jnp
import optax


def alphazero_value_loss_fn(
    pred_values: jnp.ndarray,
    target_values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the AlphaZero value loss.

    Parameters
    ----------
    pred_values : jnp.ndarray
        Predicted state values with shape ``[N, 1]``.
    target_values : jnp.ndarray
        Supervised value targets with the same shape as ``pred_values``.
    mask : jnp.ndarray
        Per-sample weights with shape ``[N]``.

    Returns
    -------
    jnp.ndarray
        Masked mean squared error over valid samples.
    """
    value_loss_per = optax.l2_loss(
        predictions=pred_values,
        targets=target_values,
    ).squeeze(-1)
    weights = mask.astype(value_loss_per.dtype)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(value_loss_per * weights) / denom


def alphazero_policy_loss_fn(
    policy_logits: jnp.ndarray,
    target_policy: jnp.ndarray,
    legal_action_mask: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the AlphaZero policy loss.

    Parameters
    ----------
    policy_logits : jnp.ndarray
        Predicted policy logits with shape ``[N, A]``.
    target_policy : jnp.ndarray
        Supervised MCTS policy targets with shape ``[N, A]``.
    legal_action_mask : jnp.ndarray
        Boolean legal-action mask with shape ``[N, A]``.
    mask : jnp.ndarray
        Per-sample weights with shape ``[N]``.

    Returns
    -------
    jnp.ndarray
        Masked cross-entropy over valid samples.
    """
    masked_policy_logits = jnp.where(legal_action_mask, policy_logits, -1e9)
    policy_loss_per = optax.softmax_cross_entropy(masked_policy_logits, target_policy)
    weights = mask.astype(policy_loss_per.dtype)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(policy_loss_per * weights) / denom

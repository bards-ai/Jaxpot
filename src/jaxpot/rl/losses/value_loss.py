import jax
import jax.numpy as jnp


def value_loss_fn(
    new_values: jnp.ndarray,
    old_values: jnp.ndarray,
    target: jnp.ndarray,
    clip_value: float = 1.0,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Clipped value loss.

    Parameters
    ----------
    new_values : jnp.ndarray
        Predicted values.
    old_values : jnp.ndarray
        Values used for clipping.
    target : jnp.ndarray
        Bootstrapped returns.
    clip_value : float
        Clip magnitude.
    mask : jnp.ndarray | None
        Optional mask for valid entries. If provided, computes a masked mean.

    Returns
    -------
    jnp.ndarray
        Scalar loss.
    """
    value_clipped = old_values + jax.lax.clamp(-clip_value, new_values - old_values, clip_value)
    value_original_loss = jnp.square(new_values - target)
    value_clipped_loss = jnp.square(value_clipped - target)
    value_loss_val = jnp.maximum(value_original_loss, value_clipped_loss)
    if mask is not None:
        weights = mask.astype(value_loss_val.dtype)
        denom = jnp.maximum(weights.sum(), 1.0)
        value_loss_val = (value_loss_val * weights).sum() / denom
    else:
        value_loss_val = value_loss_val.mean()
    return value_loss_val

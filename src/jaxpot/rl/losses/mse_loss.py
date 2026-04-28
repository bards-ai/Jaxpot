import jax.numpy as jnp


def mse_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    *,
    loss_coeff: float = 1.0,
) -> jnp.ndarray:
    """
    Weighted MSE.

    Parameters
    ----------
    pred : jnp.ndarray
        Predictions [...].
    target : jnp.ndarray
        Targets with same shape as ``pred``.
    loss_coeff : float
        Loss multiplier.

    Returns
    -------
    jnp.ndarray
        Scalar MSE loss.
    """
    err = jnp.square(pred - target)
    mse = err.mean()
    return mse * loss_coeff

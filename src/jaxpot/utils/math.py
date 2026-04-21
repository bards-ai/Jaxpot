import jax.numpy as jnp


def masked_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Mask invalid actions by setting their logits to a large negative number.

    Parameters
    ----------
    logits : jnp.ndarray
        Unmasked action logits [..., A].
    mask : jnp.ndarray
        Boolean mask of legal actions [..., A].

    Returns
    -------
    jnp.ndarray
        Masked logits with same shape as input.
    """
    return jnp.where(mask, logits, -1e9)

import jax
import jax.numpy as jnp


def policy_loss_fn(
    ratio: jnp.ndarray,
    advantages: jnp.ndarray,
    policy_loss_coeff: float = 1.0,
    clip_ratio_low: float = 0.91,
    clip_ratio_high: float = 1.1,
    normalize_advantages: bool = True,
) -> jnp.ndarray:
    """
    PPO policy loss.

    Parameters
    ----------
    ratio : jnp.ndarray
        Ratio of new policy to old policy [N].
    advantages : jnp.ndarray
        Advantage estimates [N].
    policy_loss_coeff : float
        Policy loss weight.
    clip_ratio_low : float
        Lower bound on the ratio.
    clip_ratio_high : float
        Upper bound on the ratio.
    normalize_advantages : bool
        Whether to normalize advantages.

    Returns
    -------
    jnp.ndarray
        Scalar policy loss.
    """
    if normalize_advantages:
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        advantages = (advantages - mean) / std
    clipped_ratio = jax.lax.clamp(clip_ratio_low, ratio, clip_ratio_high)
    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    loss = jnp.minimum(loss_unclipped, loss_clipped)
    loss = -loss.mean()
    loss *= policy_loss_coeff
    return loss

from __future__ import annotations

import math


def normalize_action_dim(action_dim: int | tuple[int, ...]) -> int:
    """Normalize an action shape or scalar dimension to an ``int``."""
    if isinstance(action_dim, (int, float)):
        return int(action_dim)
    return math.prod(int(dim) for dim in action_dim)


def normalize_obs_shape(obs_shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize an observation shape to a tuple of ints."""
    if isinstance(obs_shape, (list, tuple)):
        return tuple(int(dim) for dim in obs_shape)
    return (int(obs_shape),)

from __future__ import annotations

import math

from flax.nnx import nn
from jax.nn.initializers import Initializer


def orthogonal_init(scale: float = math.sqrt(2.0)) -> Initializer:
    """Return an orthogonal initializer scaled by ``scale``."""
    return nn.initializers.orthogonal(scale)


def xavier_normal_init() -> Initializer:
    """Return a Xavier/Glorot normal initializer."""
    return nn.initializers.glorot_normal()


def xavier_uniform_init() -> Initializer:
    """Return a Xavier/Glorot uniform initializer."""
    return nn.initializers.glorot_uniform()


def he_normal_init() -> Initializer:
    """Return a He normal initializer for ReLU-like activations."""
    return nn.initializers.he_normal()


def he_uniform_init() -> Initializer:
    """Return a He uniform initializer for ReLU-like activations."""
    return nn.initializers.he_uniform()

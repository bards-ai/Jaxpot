from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

from jaxpot.models.utils import he_normal_init, orthogonal_init


class MLPBlock(nnx.Module):
    """Two-layer MLP block implemented with Flax NNX."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_dim: int,
        dropout: float = 0.0,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype
        self.drop = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.fc1 = nnx.Linear(
            in_features=in_features,
            out_features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=orthogonal_init(math.sqrt(2.0)),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=orthogonal_init(math.sqrt(2.0)),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = jnn.relu(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc2(x)
        x = jnn.relu(x)
        return x


class ResidualBlock(nnx.Module):
    """Residual MLP block with pre-normalization and dropout support."""

    def __init__(
        self,
        *,
        features: int,
        dropout: float = 0.0,
        dtype: Any = jnp.float32,
        expansion: int = 4,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.dtype = dtype
        self.expansion = int(expansion)
        self.norm = nnx.LayerNorm(self.features, dtype=self.dtype, rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.fc1 = nnx.Linear(
            in_features=self.features,
            out_features=self.expansion * self.features,
            dtype=self.dtype,
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            in_features=self.expansion * self.features,
            out_features=self.features,
            dtype=self.dtype,
            kernel_init=he_normal_init(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = jnn.relu(h)
        h = self.fc2(h)
        if self.drop is not None:
            h = self.drop(h)
        return residual + h


class ConvResidualBlock(nnx.Module):
    """Pre-activation convolutional residual block."""

    def __init__(
        self,
        *,
        num_filters: int,
        dropout: float = 0.0,
        dtype: Any = jnp.float32,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.norm1 = nnx.LayerNorm(num_filters, dtype=dtype, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=num_filters,
            out_features=num_filters,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=dtype,
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_filters, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=num_filters,
            out_features=num_filters,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=dtype,
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.drop = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.drop is not None:
            h = self.drop(h)
        return residual + h

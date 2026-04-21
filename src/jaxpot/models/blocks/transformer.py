from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class TransformerBlock(nnx.Module):
    """Pre-norm transformer encoder block with self-attention and MLP."""

    def __init__(
        self, *, embed_dim: int, num_heads: int, dropout: float, expansion: int = 4, rngs: nnx.Rngs
    ):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.expansion = int(expansion)
        self.ln1 = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(embed_dim, expansion * embed_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(expansion * embed_dim, embed_dim, rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        h = self.ln1(x)
        attn_mask = (
            nnx.make_attention_mask(mask, mask, dtype=jnp.bool_) if mask is not None else None
        )
        attn_out = self.attn(h, mask=attn_mask, decode=False)
        x = x + self.drop(attn_out)
        h = self.ln2(x)
        h = self.mlp_fc1(h)
        h = nnx.gelu(h)
        h = self.drop(h)
        h = self.mlp_fc2(h)
        x = x + self.drop(h)
        return x


class AttentionPooling(nnx.Module):
    """Attention-based pooling using a learned query vector."""

    def __init__(self, *, embed_dim: int, num_heads: int, dropout: float, rngs: nnx.Rngs):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.query = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, 1, embed_dim))
        )
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.ln = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size = x.shape[0]
        query = jnp.broadcast_to(self.query.value, (batch_size, 1, self.embed_dim))
        x_norm = self.ln(x)
        attn_mask = mask[:, None, None, :].astype(jnp.bool_) if mask is not None else None
        pooled = self.attn(query, x_norm, mask=attn_mask, decode=False)
        return pooled.squeeze(1)

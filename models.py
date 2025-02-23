# models.py
import jax
import jax.numpy as jnp
from flax import linen as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (RMSNorm) without bias."""
    dim: int
    eps: float = 1e-5

    def setup(self):
        # Per-dimension scale
        self.weight = self.param('weight', lambda rng, shape: jnp.ones(shape), (self.dim,))

    def __call__(self, x):
        # RMS along last dimension
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


def apply_rope(x, base=1e6):
    """
    Apply Rotary Positional Embeddings (RoPE) to Q,K.
    x shape: [batch, seq, heads, head_dim].
    """
    b, seq, n_heads, dim = x.shape
    half = dim // 2
    if half * 2 != dim:
        raise ValueError("Head dimension must be even for RoPE.")

    # Frequencies for rotation
    i = jnp.arange(half)
    theta = 1.0 / (base ** (2 * i / dim))  # shape [half]

    # Positions
    pos = jnp.arange(seq)
    angles = pos[:, None] * theta[None, :]  # [seq, half]

    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    # We'll broadcast cos, sin to [batch, seq, heads, half]
    # Tile over batch & heads, then swapaxes back
    cos = jnp.tile(cos[None, :, None, :], (b, 1, n_heads, 1))
    sin = jnp.tile(sin[None, :, None, :], (b, 1, n_heads, 1))

    # x = [b, seq, heads, dim]
    x1, x2 = jnp.split(x, 2, axis=-1)  # each [b, seq, heads, half]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return jnp.concatenate([x1_rot, x2_rot], axis=-1)


class MultiHeadSelfAttention(nn.Module):
    dim: int
    n_heads: int
    dropout: float = 0.0

    def setup(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        self.dim_head = self.dim // self.n_heads

        self.norm = RMSNorm(self.dim)

        self.Wq = nn.Dense(self.n_heads * self.dim_head, use_bias=False)
        self.Wk = nn.Dense(self.n_heads * self.dim_head, use_bias=False)
        self.Wv = nn.Dense(self.n_heads * self.dim_head, use_bias=False)
        self.Wo = nn.Dense(self.dim, use_bias=False)

        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, training=True):
        b, seq, d = x.shape
        # Pre-norm
        x_norm = self.norm(x)

        q = self.Wq(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        k = self.Wk(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        v = self.Wv(x_norm).reshape(b, seq, self.n_heads, self.dim_head)

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # Causal mask: disallow attention to future tokens
        causal_mask = jnp.triu(
            jnp.full((seq, seq), -jnp.inf, dtype=jnp.float32), k=1
        )  # shape [seq, seq]

        # Convert to shape [b, n_heads, seq, seq] for broadcasting
        causal_mask = causal_mask[None, None, :, :]

        # Scaled dot-product
        attn_scores = jnp.einsum('bthd,bshd->bhts', q, k) / jnp.sqrt(self.dim_head)
        attn_scores = attn_scores + causal_mask
        attn_weights = nn.softmax(attn_scores, axis=-1)

        # Weighted sum
        out = jnp.einsum('bhts,bshd->bthd', attn_weights, v)
        out = out.reshape(b, seq, self.dim)
        out = self.Wo(out)
        out = self.dropout_layer(out, deterministic=not training)
        return out


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0

    def setup(self):
        self.norm = RMSNorm(self.dim)
        self.w1 = nn.Dense(self.hidden_dim, use_bias=False)
        self.w2 = nn.Dense(self.dim, use_bias=False)
        self.w3 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, training=True):
        x_norm = self.norm(x)
        x1 = self.w1(x_norm)          # [b, seq, hidden_dim]
        x_silu = nn.silu(x1)
        # Gating branch
        x2 = self.w3(x_norm)
        gated = x_silu * x2
        gated = self.dropout_layer(gated, deterministic=not training)
        out = self.w2(gated)
        return out


class Transformer(nn.Module):
    depth: int
    dim: int
    heads: int
    n_tokens: int
    seq_len: int
    dropout: float = 0.0
    pool: str = 'cls'  # 'cls' or 'mean'

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.n_tokens, features=self.dim)

        # Create a list of Transformer blocks
        self.blocks = [
            (MultiHeadSelfAttention(self.dim, self.heads, self.dropout),
             FeedForward(self.dim, 4 * self.dim, self.dropout))
            for _ in range(self.depth)
        ]

        self.final_norm = RMSNorm(self.dim)
        self.output_dense = nn.Dense(self.n_tokens, use_bias=False)

    def __call__(self, x, training=True):
        """
        x: int32 tensor [batch, seq_len].
        Returns logits [batch, n_tokens].
        """
        # Token embeddings
        x = self.embedding(x)  # [batch, seq_len, dim]

        # Transformer blocks
        for attn, ffn in self.blocks:
            x = x + attn(x, training=training)
            x = x + ffn(x, training=training)

        x = self.final_norm(x)

        # Pool
        if self.pool == 'mean':
            x = jnp.mean(x, axis=1)  # [b, dim]
        else:
            # last token
            x = x[:, -1, :]          # [b, dim]

        # Classifier
        logits = self.output_dense(x)  # [b, n_tokens]
        return logits

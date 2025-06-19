"""Unit tests for models.py module."""

import pytest
import jax
import jax.numpy as jnp
from models import RMSNorm, apply_rope, MultiHeadSelfAttention, FeedForward, Transformer


class TestRMSNorm:
    """Test cases for RMSNorm layer."""

    def test_initialization(self):
        """Test RMSNorm initialization."""
        dim = 128
        norm = RMSNorm(dim=dim)
        
        # Test with dummy input to initialize parameters
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 10, dim))
        params = norm.init(rng, x)
        
        assert 'params' in params
        assert 'weight' in params['params']
        assert params['params']['weight'].shape == (dim,)

    def test_forward_pass(self):
        """Test RMSNorm forward pass."""
        dim = 64
        batch_size = 2
        seq_len = 5
        
        norm = RMSNorm(dim=dim)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (batch_size, seq_len, dim))
        
        params = norm.init(rng, x)
        output = norm.apply(params, x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that normalization is applied (approximate RMS should be close to 1)
        rms = jnp.sqrt(jnp.mean(output**2, axis=-1))
        assert jnp.allclose(rms, 1.0, atol=1e-5)


class TestApplyRope:
    """Test cases for RoPE (Rotary Position Embeddings)."""

    def test_rope_shape_preservation(self):
        """Test that RoPE preserves input shape."""
        batch_size, seq_len, n_heads, head_dim = 2, 4, 2, 8
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (batch_size, seq_len, n_heads, head_dim))
        
        output = apply_rope(x)
        assert output.shape == x.shape

    def test_rope_even_dimension_requirement(self):
        """Test that RoPE requires even head dimension."""
        batch_size, seq_len, n_heads, head_dim = 2, 4, 2, 7  # Odd dimension
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (batch_size, seq_len, n_heads, head_dim))
        
        with pytest.raises(ValueError, match="Head dimension must be even"):
            apply_rope(x)

    def test_rope_different_positions(self):
        """Test that RoPE produces different outputs for different positions."""
        batch_size, seq_len, n_heads, head_dim = 1, 3, 1, 4
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (batch_size, seq_len, n_heads, head_dim))
        
        # Make all positions have the same values
        x = jnp.tile(x[:, :1, :, :], (1, seq_len, 1, 1))
        
        output = apply_rope(x)
        
        # After RoPE, different positions should have different values
        assert not jnp.allclose(output[:, 0, :, :], output[:, 1, :, :])
        assert not jnp.allclose(output[:, 1, :, :], output[:, 2, :, :])


class TestMultiHeadSelfAttention:
    """Test cases for MultiHeadSelfAttention layer."""

    def test_attention_initialization(self):
        """Test attention layer initialization."""
        dim = 128
        n_heads = 4
        dropout = 0.1
        
        attn = MultiHeadSelfAttention(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Test initialization
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 10, dim))
        params = attn.init(rng, x, training=True)
        
        assert 'params' in params
        
        # Check parameter shapes
        expected_qkv_shape = (dim, dim)  # Since n_heads * dim_head = dim
        assert params['params']['Wq']['kernel'].shape == expected_qkv_shape
        assert params['params']['Wk']['kernel'].shape == expected_qkv_shape
        assert params['params']['Wv']['kernel'].shape == expected_qkv_shape
        assert params['params']['Wo']['kernel'].shape == expected_qkv_shape

    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        dim = 64
        n_heads = 4
        batch_size = 2
        seq_len = 8
        
        attn = MultiHeadSelfAttention(dim=dim, n_heads=n_heads, dropout=0.0)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (batch_size, seq_len, dim))
        
        params = attn.init(rng, x, training=True)
        output = attn.apply(params, x, training=True, rngs={'dropout': rng})
        
        # Check output shape
        assert output.shape == x.shape

    def test_attention_causal_mask(self):
        """Test that causal masking works correctly."""
        dim = 32
        n_heads = 2
        seq_len = 4
        
        attn = MultiHeadSelfAttention(dim=dim, n_heads=n_heads, dropout=0.0)
        rng = jax.random.PRNGKey(0)
        
        # Create input where all positions are identical
        x = jnp.ones((1, seq_len, dim))
        params = attn.init(rng, x, training=True)
        
        # Forward pass
        output = attn.apply(params, x, training=False)
        
        # The output should still be finite (no NaN from masked attention)
        assert jnp.all(jnp.isfinite(output))


class TestFeedForward:
    """Test cases for FeedForward layer."""

    def test_feedforward_initialization(self):
        """Test FeedForward initialization."""
        dim = 128
        hidden_dim = 512
        dropout = 0.1
        
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 10, dim))
        params = ff.init(rng, x, training=True)
        
        assert 'params' in params
        assert params['params']['w1']['kernel'].shape == (dim, hidden_dim)
        assert params['params']['w2']['kernel'].shape == (hidden_dim, dim)
        assert params['params']['w3']['kernel'].shape == (dim, hidden_dim)

    def test_feedforward_forward_pass(self):
        """Test FeedForward forward pass."""
        dim = 64
        hidden_dim = 256
        batch_size = 2
        seq_len = 5
        
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.0)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (batch_size, seq_len, dim))
        
        params = ff.init(rng, x, training=True)
        output = ff.apply(params, x, training=True, rngs={'dropout': rng})
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that SiLU activation is working (output should not be same as input)
        assert not jnp.allclose(output, x)


class TestTransformer:
    """Test cases for complete Transformer model."""

    def test_transformer_initialization(self):
        """Test Transformer initialization."""
        depth = 2
        dim = 64
        heads = 4
        n_tokens = 100
        seq_len = 10
        dropout = 0.1
        
        model = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=dropout, pool='cls'
        )
        
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, seq_len), dtype=jnp.int32)
        params = model.init(rng, x, training=True)
        
        assert 'params' in params
        
        # Check embedding layer
        assert params['params']['embedding']['embedding'].shape == (n_tokens, dim)
        
        # Check output layer
        assert params['params']['output_dense']['kernel'].shape == (dim, n_tokens)

    def test_transformer_forward_pass(self):
        """Test Transformer forward pass."""
        depth = 1
        dim = 32
        heads = 2
        n_tokens = 50
        seq_len = 6
        batch_size = 3
        
        model = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        
        rng = jax.random.PRNGKey(42)
        # Create valid token indices
        x = jax.random.randint(rng, (batch_size, seq_len), 0, n_tokens, dtype=jnp.int32)
        
        params = model.init(rng, x, training=True)
        output = model.apply(params, x, training=True, rngs={'dropout': rng})
        
        # Check output shape - should be [batch_size, n_tokens] for classification
        assert output.shape == (batch_size, n_tokens)
        
        # Check that output is finite
        assert jnp.all(jnp.isfinite(output))

    def test_transformer_different_pooling(self):
        """Test Transformer with different pooling strategies."""
        depth = 1
        dim = 32
        heads = 2
        n_tokens = 50
        seq_len = 6
        batch_size = 2
        
        rng = jax.random.PRNGKey(0)
        x = jax.random.randint(rng, (batch_size, seq_len), 0, n_tokens, dtype=jnp.int32)
        
        # Test 'cls' pooling (last token)
        model_cls = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        params_cls = model_cls.init(rng, x, training=True)
        output_cls = model_cls.apply(params_cls, x, training=False)
        
        # Test 'mean' pooling
        model_mean = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='mean'
        )
        params_mean = model_mean.init(rng, x, training=True)
        output_mean = model_mean.apply(params_mean, x, training=False)
        
        # Both should have same output shape
        assert output_cls.shape == output_mean.shape == (batch_size, n_tokens)
        
        # Outputs should be different due to different pooling strategies
        assert not jnp.allclose(output_cls, output_mean)

    def test_transformer_training_vs_inference(self):
        """Test Transformer behavior in training vs inference mode."""
        depth = 1
        dim = 32
        heads = 2
        n_tokens = 30
        seq_len = 4
        batch_size = 2
        
        model = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.5, pool='cls'  # High dropout to see difference
        )
        
        rng = jax.random.PRNGKey(0)
        x = jax.random.randint(rng, (batch_size, seq_len), 0, n_tokens, dtype=jnp.int32)
        
        params = model.init(rng, x, training=True)
        
        # Training mode (with dropout)
        rng1, rng2 = jax.random.split(rng)
        output_train1 = model.apply(params, x, training=True, rngs={'dropout': rng1})
        output_train2 = model.apply(params, x, training=True, rngs={'dropout': rng2})
        
        # Inference mode (without dropout)
        output_inference = model.apply(params, x, training=False)
        
        # Training outputs with different RNG should be different (due to dropout)
        assert not jnp.allclose(output_train1, output_train2)
        
        # All outputs should have correct shape
        expected_shape = (batch_size, n_tokens)
        assert output_train1.shape == expected_shape
        assert output_train2.shape == expected_shape
        assert output_inference.shape == expected_shape

    def test_transformer_dimension_compatibility(self):
        """Test that dimension parameters are compatible."""
        # Test invalid head configuration
        with pytest.raises(AssertionError, match="dim must be divisible by n_heads"):
            dim = 65  # Not divisible by n_heads=4
            heads = 4
            
            model = Transformer(
                depth=1, dim=dim, heads=heads,
                n_tokens=50, seq_len=10,
                dropout=0.0, pool='cls'
            )
            
            rng = jax.random.PRNGKey(0)
            x = jnp.ones((1, 10), dtype=jnp.int32)
            model.init(rng, x, training=True)

    def test_transformer_residual_connections(self):
        """Test that residual connections are working."""
        depth = 2
        dim = 32
        heads = 2
        n_tokens = 30
        seq_len = 4
        batch_size = 1
        
        model = Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        
        rng = jax.random.PRNGKey(0)
        x = jax.random.randint(rng, (batch_size, seq_len), 0, n_tokens, dtype=jnp.int32)
        
        params = model.init(rng, x, training=True)
        output = model.apply(params, x, training=False)
        
        # Should produce finite outputs even with multiple layers
        assert jnp.all(jnp.isfinite(output))
        assert output.shape == (batch_size, n_tokens)
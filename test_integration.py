"""Integration tests for the complete grokking system."""

import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state

import data
import models


class TestGrokkingIntegration:
    """Integration tests for the complete grokking pipeline."""

    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline with small model and data."""
        # Small configuration for fast testing
        p = 7
        operation = '+'
        train_fraction = 0.7
        
        depth = 1
        dim = 32
        heads = 2
        dropout = 0.0
        
        epochs = 3
        batch_size = 8
        learning_rate = 1e-3
        weight_decay = 0.1
        seed = 42
        
        # 1. Generate data
        X_train, y_train, X_val, y_val = data.grokking_data(
            p, op=operation, train_fraction=train_fraction, seed=seed
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        # 2. Create model
        model = models.Transformer(
            depth=depth, dim=dim, heads=heads,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=dropout, pool='cls'
        )
        
        # 3. Initialize parameters
        rng = jax.random.PRNGKey(seed)
        init_batch = X_train[:1]
        params = model.init(rng, init_batch, training=True)['params']
        
        # 4. Create optimizer
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # 5. Create train state
        class TrainState(train_state.TrainState):
            pass
        
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        
        # 6. Define functions
        def loss_fn(params, X, y, rng_key):
            logits = model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
            one_hot = jax.nn.one_hot(y, n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss
        
        @jax.jit
        def train_step(state, X, y, rng_key):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, X, y, rng_key)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            new_state = state.replace(params=new_params, opt_state=new_opt_state)
            return new_state, loss
        
        @jax.jit
        def eval_step(params, X, y):
            logits = model.apply({'params': params}, X, training=False)
            one_hot = jax.nn.one_hot(y, n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean((preds == y).astype(jnp.float32))
            return loss, acc
        
        # 7. Training loop
        num_train = X_train.shape[0]
        num_batches = int(np.ceil(num_train / batch_size))
        
        initial_val_loss, initial_val_acc = eval_step(state.params, X_val, y_val)
        
        for epoch in range(1, epochs + 1):
            # Shuffle data
            perm = np.random.permutation(num_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            epoch_loss = 0.0
            for i in range(num_batches):
                batch_X = X_train_shuffled[i * batch_size : (i+1) * batch_size]
                batch_y = y_train_shuffled[i * batch_size : (i+1) * batch_size]
                
                rng, dropout_key = jax.random.split(rng)
                state, loss_val = train_step(state, batch_X, batch_y, dropout_key)
                epoch_loss += float(loss_val) * batch_X.shape[0]
            
            train_loss = epoch_loss / num_train
            val_loss, val_acc = eval_step(state.params, X_val, y_val)
            
            # Check that values are finite
            assert jnp.isfinite(train_loss)
            assert jnp.isfinite(val_loss)
            assert jnp.isfinite(val_acc)
            
            # Check that accuracy is in valid range
            assert 0 <= val_acc <= 1
        
        # Check that training made some progress (loss should decrease or accuracy improve)
        final_val_loss, final_val_acc = eval_step(state.params, X_val, y_val)
        
        # Either loss decreased or accuracy improved (or both)
        improved = (final_val_loss < initial_val_loss) or (final_val_acc > initial_val_acc)
        assert improved, f"Training should improve performance. Initial: loss={initial_val_loss:.4f}, acc={initial_val_acc:.4f}. Final: loss={final_val_loss:.4f}, acc={final_val_acc:.4f}"

    def test_different_operations_training(self):
        """Test training with different arithmetic operations."""
        operations = ['*', '/', '+', '-']
        
        # Small configuration
        p = 5
        depth = 1
        dim = 16
        heads = 2
        epochs = 2
        seed = 42
        
        for operation in operations:
            # Generate data
            X_train, y_train, X_val, y_val = data.grokking_data(
                p, op=operation, train_fraction=0.8, seed=seed
            )
            seq_len = X_train.shape[1]
            n_tokens = p + 2
            
            # Create and initialize model
            model = models.Transformer(
                depth=depth, dim=dim, heads=heads,
                n_tokens=n_tokens, seq_len=seq_len,
                dropout=0.0, pool='cls'
            )
            
            rng = jax.random.PRNGKey(seed)
            init_batch = X_train[:1]
            params = model.init(rng, init_batch, training=True)['params']
            
            # Quick training test
            optimizer = optax.adamw(learning_rate=1e-3)
            
            class TrainState(train_state.TrainState):
                pass
            
            state = TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optimizer
            )
            
            def loss_fn(params, X, y, rng_key):
                logits = model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
                one_hot = jax.nn.one_hot(y, n_tokens)
                loss = optax.softmax_cross_entropy(logits, one_hot).mean()
                return loss
            
            # Test one training step
            batch_X = X_train[:4]
            batch_y = y_train[:4]
            rng_key = jax.random.PRNGKey(0)
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, batch_X, batch_y, rng_key)
            
            # Check that loss and gradients are finite
            assert jnp.isfinite(loss), f"Loss should be finite for operation {operation}"
            
            def check_finite_grads(grad_dict):
                for key, value in grad_dict.items():
                    if isinstance(value, dict):
                        check_finite_grads(value)
                    else:
                        assert jnp.all(jnp.isfinite(value)), f"Gradients should be finite for operation {operation}"
            
            check_finite_grads(grads)

    def test_model_size_scaling(self):
        """Test that the system works with different model sizes."""
        configurations = [
            {'depth': 1, 'dim': 16, 'heads': 1},
            {'depth': 2, 'dim': 32, 'heads': 2},
            {'depth': 1, 'dim': 64, 'heads': 4},
        ]
        
        p = 5
        operation = '+'
        seed = 42
        
        # Generate data once
        X_train, y_train, X_val, y_val = data.grokking_data(
            p, op=operation, train_fraction=0.7, seed=seed
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        for config in configurations:
            # Create model with current configuration
            model = models.Transformer(
                depth=config['depth'],
                dim=config['dim'],
                heads=config['heads'],
                n_tokens=n_tokens,
                seq_len=seq_len,
                dropout=0.0,
                pool='cls'
            )
            
            # Initialize and test forward pass
            rng = jax.random.PRNGKey(seed)
            init_batch = X_train[:1]
            params = model.init(rng, init_batch, training=True)['params']
            
            # Test forward pass
            batch_X = X_train[:3]
            logits = model.apply({'params': params}, batch_X, training=False)
            
            assert logits.shape == (3, n_tokens)
            assert jnp.all(jnp.isfinite(logits))

    def test_batch_size_handling(self):
        """Test that the system handles different batch sizes correctly."""
        p = 5
        operation = '+'
        
        # Generate data
        X_train, y_train, X_val, y_val = data.grokking_data(
            p, op=operation, train_fraction=0.8, seed=42
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        # Create model
        model = models.Transformer(
            depth=1, dim=16, heads=2,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        
        rng = jax.random.PRNGKey(42)
        init_batch = X_train[:1]
        params = model.init(rng, init_batch, training=True)['params']
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5, len(X_train)]  # Including full dataset
        
        for batch_size in batch_sizes:
            if batch_size <= len(X_train):
                batch_X = X_train[:batch_size]
                batch_y = y_train[:batch_size]
                
                # Test forward pass
                logits = model.apply({'params': params}, batch_X, training=False)
                assert logits.shape == (batch_size, n_tokens)
                assert jnp.all(jnp.isfinite(logits))
                
                # Test loss computation
                one_hot = jax.nn.one_hot(batch_y, n_tokens)
                loss = optax.softmax_cross_entropy(logits, one_hot).mean()
                assert jnp.isfinite(loss)

    def test_reproducibility_across_runs(self):
        """Test that identical configurations produce identical results."""
        p = 7
        operation = '*'
        seed = 123
        
        def run_training():
            # Generate data
            X_train, y_train, X_val, y_val = data.grokking_data(
                p, op=operation, train_fraction=0.6, seed=seed
            )
            seq_len = X_train.shape[1]
            n_tokens = p + 2
            
            # Create model
            model = models.Transformer(
                depth=1, dim=16, heads=2,
                n_tokens=n_tokens, seq_len=seq_len,
                dropout=0.0, pool='cls'
            )
            
            # Initialize
            rng = jax.random.PRNGKey(seed)
            init_batch = X_train[:1]
            params = model.init(rng, init_batch, training=True)['params']
            
            # One forward pass
            logits = model.apply({'params': params}, X_train[:3], training=False)
            return logits
        
        # Run twice with identical settings
        result1 = run_training()
        result2 = run_training()
        
        # Results should be identical
        assert jnp.allclose(result1, result2)

    def test_gradient_flow_through_model(self):
        """Test that gradients flow through all parts of the model."""
        p = 5
        operation = '+'
        
        # Generate data
        X_train, y_train, _, _ = data.grokking_data(
            p, op=operation, train_fraction=1.0, seed=42
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        # Create model
        model = models.Transformer(
            depth=2, dim=32, heads=2,  # Multi-layer to test gradient flow
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        
        rng = jax.random.PRNGKey(42)
        init_batch = X_train[:1]
        params = model.init(rng, init_batch, training=True)['params']
        
        # Define loss
        def loss_fn(params):
            batch_X = X_train[:3]
            batch_y = y_train[:3]
            logits = model.apply({'params': params}, batch_X, training=False)
            one_hot = jax.nn.one_hot(batch_y, n_tokens)
            return optax.softmax_cross_entropy(logits, one_hot).mean()
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Check that all parameters have non-zero gradients
        # (This ensures gradients flow through the entire model)
        def check_grad_flow(grad_dict, path=""):
            for param_name, param_grad in grad_dict.items():
                current_path = f"{path}.{param_name}" if path else param_name
                if isinstance(param_grad, dict):
                    check_grad_flow(param_grad, current_path)
                else:
                    grad_norm = jnp.linalg.norm(param_grad)
                    assert grad_norm > 0, f"Gradient for {current_path} should be non-zero"
        
        check_grad_flow(grads)

    def test_memory_efficiency_large_batch(self):
        """Test that the system can handle reasonably large batches without memory issues."""
        p = 11  # Larger prime for more data
        operation = '/'
        
        # Generate larger dataset
        X_train, y_train, X_val, y_val = data.grokking_data(
            p, op=operation, train_fraction=0.8, seed=42
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        # Create model
        model = models.Transformer(
            depth=1, dim=32, heads=4,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.0, pool='cls'
        )
        
        rng = jax.random.PRNGKey(42)
        init_batch = X_train[:1]
        params = model.init(rng, init_batch, training=True)['params']
        
        # Test with largest possible batch (full training set)
        logits = model.apply({'params': params}, X_train, training=False)
        
        assert logits.shape == (len(X_train), n_tokens)
        assert jnp.all(jnp.isfinite(logits))
        
        # Test evaluation on full validation set
        val_logits = model.apply({'params': params}, X_val, training=False)
        val_preds = jnp.argmax(val_logits, axis=-1)
        val_acc = jnp.mean((val_preds == y_val).astype(jnp.float32))
        
        assert jnp.isfinite(val_acc)
        assert 0 <= val_acc <= 1

    def test_model_state_consistency(self):
        """Test that model state remains consistent between training and evaluation."""
        p = 5
        operation = '+'
        
        X_train, y_train, X_val, y_val = data.grokking_data(
            p, op=operation, train_fraction=0.7, seed=42
        )
        seq_len = X_train.shape[1]
        n_tokens = p + 2
        
        # Create model with dropout for testing state consistency
        model = models.Transformer(
            depth=1, dim=16, heads=2,
            n_tokens=n_tokens, seq_len=seq_len,
            dropout=0.1, pool='cls'
        )
        
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, X_train[:1], training=True)['params']
        
        batch_X = X_train[:3]
        
        # Multiple inference runs should give the same result (no dropout in eval mode)
        output1 = model.apply({'params': params}, batch_X, training=False)
        output2 = model.apply({'params': params}, batch_X, training=False)
        
        assert jnp.allclose(output1, output2)
        
        # Training mode with same RNG should give same result
        rng1 = jax.random.PRNGKey(999)
        rng2 = jax.random.PRNGKey(999)
        
        output_train1 = model.apply({'params': params}, batch_X, training=True, rngs={'dropout': rng1})
        output_train2 = model.apply({'params': params}, batch_X, training=True, rngs={'dropout': rng2})
        
        assert jnp.allclose(output_train1, output_train2)
        
        # Training mode with different RNG should give different results (due to dropout)
        rng3 = jax.random.PRNGKey(888)
        output_train3 = model.apply({'params': params}, batch_X, training=True, rngs={'dropout': rng3})
        
        assert not jnp.allclose(output_train1, output_train3)
"""Unit tests for main.py functions."""

import pytest
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import data
import models


class TestMainFunctions:
    """Test cases for functions from main.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.p = 5
        self.operation = '+'
        self.seed = 42
        
        # Generate small test data
        self.X_train, self.y_train, self.X_val, self.y_val = data.grokking_data(
            self.p, op=self.operation, train_fraction=0.6, seed=self.seed
        )
        self.seq_len = self.X_train.shape[1]
        self.n_tokens = self.p + 2
        
        # Create small model for testing
        self.model = models.Transformer(
            depth=1, dim=16, heads=2,
            n_tokens=self.n_tokens, seq_len=self.seq_len,
            dropout=0.0, pool='cls'
        )
        
        # Initialize model parameters
        self.rng = jax.random.PRNGKey(self.seed)
        init_batch = self.X_train[:1]
        self.params = self.model.init(self.rng, init_batch, training=True)['params']

    def test_loss_function(self):
        """Test the loss computation function."""
        # Define loss function (from main.py)
        def loss_fn(params, X, y, rng_key):
            logits = self.model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
            one_hot = jax.nn.one_hot(y, self.n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss
        
        # Test with single batch
        batch_X = self.X_train[:2]
        batch_y = self.y_train[:2]
        rng_key = jax.random.PRNGKey(0)
        
        loss = loss_fn(self.params, batch_X, batch_y, rng_key)
        
        # Check that loss is a scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0  # Cross-entropy loss should be non-negative

    def test_eval_function(self):
        """Test the evaluation function."""
        # Define eval function (from main.py)
        @jax.jit
        def eval_step(params, X, y):
            logits = self.model.apply({'params': params}, X, training=False)
            one_hot = jax.nn.one_hot(y, self.n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean((preds == y).astype(jnp.float32))
            return loss, acc
        
        # Test evaluation
        val_loss, val_acc = eval_step(self.params, self.X_val, self.y_val)
        
        # Check return types and values
        assert val_loss.shape == ()
        assert val_acc.shape == ()
        assert jnp.isfinite(val_loss)
        assert jnp.isfinite(val_acc)
        assert val_loss >= 0
        assert 0 <= val_acc <= 1  # Accuracy should be between 0 and 1

    def test_train_step(self):
        """Test the training step function."""
        # Create optimizer
        optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.1)
        
        # Create train state
        class TrainState(train_state.TrainState):
            pass
        
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=optimizer
        )
        
        # Define loss and train step functions
        def loss_fn(params, X, y, rng_key):
            logits = self.model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
            one_hot = jax.nn.one_hot(y, self.n_tokens)
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
        
        # Test training step
        batch_X = self.X_train[:2]
        batch_y = self.y_train[:2]
        rng_key = jax.random.PRNGKey(123)
        
        original_params = state.params
        new_state, train_loss = train_step(state, batch_X, batch_y, rng_key)
        
        # Check that parameters were updated
        def params_equal(p1, p2):
            return all(jnp.allclose(p1[k], p2[k]) if isinstance(p1[k], jnp.ndarray) 
                      else params_equal(p1[k], p2[k]) for k in p1)
        
        param_changed = not params_equal(original_params, new_state.params)
        assert param_changed, "Parameters should have been updated"
        
        # Check loss value
        assert train_loss.shape == ()
        assert jnp.isfinite(train_loss)
        assert train_loss >= 0

    def test_learning_rate_schedule(self):
        """Test learning rate schedule creation."""
        learning_rate = 1e-3
        warmup_steps = 5
        
        # Create schedule (from main.py)
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps
        )
        constant_fn = optax.constant_schedule(value=learning_rate)
        schedule_fn = optax.join_schedules(
            [warmup_fn, constant_fn],
            boundaries=[warmup_steps]
        )
        
        # Test schedule at different steps
        assert schedule_fn(0) == 0.0  # Should start at 0
        assert schedule_fn(warmup_steps) == learning_rate  # Should reach target at warmup_steps
        assert schedule_fn(warmup_steps + 10) == learning_rate  # Should stay constant after warmup

    def test_optimizer_creation(self):
        """Test optimizer creation with parameters."""
        learning_rate = 1e-3
        weight_decay = 0.5
        beta1 = 0.9
        beta2 = 0.98
        
        # Create optimizer (from main.py)
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # Test that optimizer can be used
        opt_state = optimizer.init(self.params)
        
        # Simulate a gradient update
        fake_grads = jax.tree.map(jnp.ones_like, self.params)
        updates, new_opt_state = optimizer.update(fake_grads, opt_state, self.params)
        
        # Check that updates were computed
        assert updates is not None
        assert new_opt_state is not None

    def test_model_initialization_reproducibility(self):
        """Test that model initialization is reproducible."""
        rng1 = jax.random.PRNGKey(42)
        rng2 = jax.random.PRNGKey(42)
        
        init_batch = self.X_train[:1]
        
        params1 = self.model.init(rng1, init_batch, training=True)['params']
        params2 = self.model.init(rng2, init_batch, training=True)['params']
        
        # Parameters should be identical with same seed
        def params_equal(p1, p2):
            return all(jnp.allclose(p1[k], p2[k]) if isinstance(p1[k], jnp.ndarray) 
                      else params_equal(p1[k], p2[k]) for k in p1)
        
        assert params_equal(params1, params2)

    def test_batch_processing(self):
        """Test processing of different batch sizes."""
        # Test with single sample
        batch_X = self.X_train[:1]
        batch_y = self.y_train[:1]
        
        logits = self.model.apply({'params': self.params}, batch_X, training=False)
        assert logits.shape == (1, self.n_tokens)
        
        # Test with multiple samples
        batch_X = self.X_train[:3]
        batch_y = self.y_train[:3]
        
        logits = self.model.apply({'params': self.params}, batch_X, training=False)
        assert logits.shape == (3, self.n_tokens)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        def loss_fn(params, X, y, rng_key):
            logits = self.model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
            one_hot = jax.nn.one_hot(y, self.n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss
        
        batch_X = self.X_train[:2]
        batch_y = self.y_train[:2]
        rng_key = jax.random.PRNGKey(0)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.params, batch_X, batch_y, rng_key)
        
        # Check that gradients have same structure as parameters
        assert grads.keys() == self.params.keys()
        
        # Check that gradients are finite
        def check_finite_grads(grad_dict):
            for key, value in grad_dict.items():
                if isinstance(value, dict):
                    check_finite_grads(value)
                else:
                    assert jnp.all(jnp.isfinite(value)), f"Gradients should be finite for key {key}"
        
        check_finite_grads(grads)
        
        # Check that at least some gradients are non-zero
        total_grad_norm = sum(jnp.sum(jnp.abs(g)) for g in jax.tree.leaves(grads))
        assert total_grad_norm > 0, "Gradients should not all be zero"

    def test_different_operations_compatibility(self):
        """Test that the model works with different arithmetic operations."""
        operations = ['*', '/', '+', '-']
        
        for op in operations:
            # Generate data for this operation
            X_train, y_train, X_val, y_val = data.grokking_data(
                self.p, op=op, train_fraction=0.6, seed=self.seed
            )
            
            # Test that model can process this data
            batch_X = X_train[:2]
            batch_y = y_train[:2]
            
            logits = self.model.apply({'params': self.params}, batch_X, training=False)
            assert logits.shape == (2, self.n_tokens)
            assert jnp.all(jnp.isfinite(logits))

    def test_training_determinism_with_seed(self):
        """Test that training is deterministic when using the same random seed."""
        # Set up identical training scenarios
        optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.1)
        
        class TrainState(train_state.TrainState):
            pass
        
        def loss_fn(params, X, y, rng_key):
            logits = self.model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
            one_hot = jax.nn.one_hot(y, self.n_tokens)
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
        
        # Create two identical states
        state1 = TrainState.create(apply_fn=self.model.apply, params=self.params, tx=optimizer)
        state2 = TrainState.create(apply_fn=self.model.apply, params=self.params, tx=optimizer)
        
        # Perform identical training steps
        batch_X = self.X_train[:2]
        batch_y = self.y_train[:2]
        rng_key = jax.random.PRNGKey(999)
        
        new_state1, loss1 = train_step(state1, batch_X, batch_y, rng_key)
        new_state2, loss2 = train_step(state2, batch_X, batch_y, rng_key)
        
        # Results should be identical
        assert jnp.allclose(loss1, loss2)
        def params_equal(p1, p2):
            return all(jnp.allclose(p1[k], p2[k]) if isinstance(p1[k], jnp.ndarray) 
                      else params_equal(p1[k], p2[k]) for k in p1)
        
        assert params_equal(new_state1.params, new_state2.params)
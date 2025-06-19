"""Unit tests for data.py module."""

import pytest
import numpy as np
import jax.numpy as jnp
from data import grokking_data


class TestGrokkingData:
    """Test cases for the grokking_data function."""

    def test_basic_functionality(self):
        """Test basic data generation functionality."""
        p = 5
        X_train, y_train, X_test, y_test = grokking_data(p, op='*', train_fraction=0.5, seed=42)
        
        # Check return types
        assert isinstance(X_train, jnp.ndarray)
        assert isinstance(y_train, jnp.ndarray)
        assert isinstance(X_test, jnp.ndarray)
        assert isinstance(y_test, jnp.ndarray)
        
        # Check data types
        assert X_train.dtype == jnp.int32
        assert y_train.dtype == jnp.int32
        assert X_test.dtype == jnp.int32
        assert y_test.dtype == jnp.int32

    def test_sequence_format(self):
        """Test that sequences are formatted correctly."""
        p = 5
        X_train, y_train, X_test, y_test = grokking_data(p, op='+', train_fraction=0.6, seed=42)
        
        # Check sequence length (should be 4: a, op_token, b, eq_token)
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4
        
        # Check that op_token and eq_token are correct
        op_token = p      # ID for the operation
        eq_token = p + 1  # ID for '='
        
        # All sequences should have op_token in position 1 and eq_token in position 3
        assert jnp.all(X_train[:, 1] == op_token)
        assert jnp.all(X_train[:, 3] == eq_token)
        assert jnp.all(X_test[:, 1] == op_token)
        assert jnp.all(X_test[:, 3] == eq_token)

    def test_train_test_split(self):
        """Test that train/test split works correctly."""
        p = 7
        train_fraction = 0.7
        X_train, y_train, X_test, y_test = grokking_data(p, op='-', train_fraction=train_fraction, seed=42)
        
        total_samples = len(X_train) + len(X_test)
        expected_train_size = int(train_fraction * total_samples)
        
        # Allow for rounding differences
        assert abs(len(X_train) - expected_train_size) <= 1
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_different_operations(self):
        """Test all supported operations."""
        p = 11
        operations = ['*', '/', '+', '-']
        
        for op in operations:
            X_train, y_train, X_test, y_test = grokking_data(p, op=op, train_fraction=0.5, seed=42)
            
            # Check that data is generated
            assert len(X_train) > 0
            assert len(y_train) > 0
            assert len(X_test) > 0
            assert len(y_test) > 0
            
            # Check value ranges
            assert jnp.all(X_train[:, 0] >= 0) and jnp.all(X_train[:, 0] < p)  # a values
            assert jnp.all(X_train[:, 2] >= 0) and jnp.all(X_train[:, 2] < p)  # b values
            assert jnp.all(y_train >= 0) and jnp.all(y_train < p)  # results

    def test_division_excludes_zero_divisor(self):
        """Test that division operation excludes b=0."""
        p = 7
        X_train, y_train, X_test, y_test = grokking_data(p, op='/', train_fraction=0.5, seed=42)
        
        # For division, b (position 2) should never be 0
        assert jnp.all(X_train[:, 2] > 0)
        assert jnp.all(X_test[:, 2] > 0)

    def test_multiplication_includes_zero(self):
        """Test that multiplication includes b=0."""
        p = 7
        X_train, y_train, X_test, y_test = grokking_data(p, op='*', train_fraction=0.5, seed=42)
        
        # For multiplication, b can be 0
        b_values = jnp.concatenate([X_train[:, 2], X_test[:, 2]])
        assert jnp.any(b_values == 0)

    def test_arithmetic_correctness(self):
        """Test that arithmetic operations are computed correctly."""
        p = 5
        
        # Test addition
        X_train, y_train, _, _ = grokking_data(p, op='+', train_fraction=1.0, seed=42)
        for i in range(min(10, len(X_train))):  # Test first 10 samples
            a, b = int(X_train[i, 0]), int(X_train[i, 2])
            expected = (a + b) % p
            assert int(y_train[i]) == expected
        
        # Test multiplication
        X_train, y_train, _, _ = grokking_data(p, op='*', train_fraction=1.0, seed=42)
        for i in range(min(10, len(X_train))):
            a, b = int(X_train[i, 0]), int(X_train[i, 2])
            expected = (a * b) % p
            assert int(y_train[i]) == expected

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        p = 7
        seed = 123
        
        X1, y1, X_test1, y_test1 = grokking_data(p, op='+', train_fraction=0.5, seed=seed)
        X2, y2, X_test2, y_test2 = grokking_data(p, op='+', train_fraction=0.5, seed=seed)
        
        # Results should be identical
        assert jnp.array_equal(X1, X2)
        assert jnp.array_equal(y1, y2)
        assert jnp.array_equal(X_test1, X_test2)
        assert jnp.array_equal(y_test1, y_test2)

    def test_invalid_operation(self):
        """Test that invalid operations raise ValueError."""
        p = 5
        with pytest.raises(ValueError, match="Unsupported operation"):
            grokking_data(p, op='%', train_fraction=0.5, seed=42)
        
        with pytest.raises(ValueError, match="Unsupported operation"):
            grokking_data(p, op='invalid', train_fraction=0.5, seed=42)

    def test_edge_cases(self):
        """Test edge cases for parameters."""
        # Small prime
        p = 2
        X_train, y_train, X_test, y_test = grokking_data(p, op='+', train_fraction=0.5, seed=42)
        assert len(X_train) > 0 or len(X_test) > 0
        
        # Train fraction = 0 (all test)
        X_train, y_train, X_test, y_test = grokking_data(5, op='+', train_fraction=0.0, seed=42)
        assert len(X_train) == 0
        assert len(X_test) > 0
        
        # Train fraction = 1 (all train)
        X_train, y_train, X_test, y_test = grokking_data(5, op='+', train_fraction=1.0, seed=42)
        assert len(X_train) > 0
        assert len(X_test) == 0

    def test_token_encoding(self):
        """Test that tokens are encoded correctly."""
        p = 7
        X_train, y_train, X_test, y_test = grokking_data(p, op='*', train_fraction=0.5, seed=42)
        
        op_token = p      # Should be 7
        eq_token = p + 1  # Should be 8
        
        # Check that op_token and eq_token are in correct positions
        assert jnp.all(X_train[:, 1] == op_token)
        assert jnp.all(X_train[:, 3] == eq_token)
        assert jnp.all(X_test[:, 1] == op_token)
        assert jnp.all(X_test[:, 3] == eq_token)
        
        # Check that operands are in valid range
        assert jnp.all(X_train[:, 0] < p)  # a < p
        assert jnp.all(X_train[:, 2] < p)  # b < p
        assert jnp.all(y_train < p)        # result < p
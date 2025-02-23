# data.py
import numpy as np
import jax.numpy as jnp

def grokking_data(p: int, op: str = '/', train_fraction: float = 0.5, seed: int = 42):
    """
    Generate training and test data for modular arithmetic grokking.
    - p: prime modulus for arithmetic (e.g., 97)
    - op: operation in {'*','/','+','-'} (defaults to '/')
    - train_fraction: fraction of data used for training (remainder is validation)
    - seed: random seed
    Returns: X_train, y_train, X_test, y_test as JAX arrays (int32).
    """
    # Supported operations (results mod p)
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p - 2, p)) % p,  # Fermat's little theorem
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }
    if op not in operations:
        raise ValueError("Unsupported operation. Choose from ['*', '/', '+', '-'].")

    # Generate all pairs (a, b), excluding b=0 for division
    b_start = 1 if op == '/' else 0
    pairs = [(a, b) for a in range(p) for b in range(b_start, p)]
    results = [operations[op](a, b) for (a, b) in pairs]

    pairs = np.array(pairs, dtype=int)
    results = np.array(results, dtype=int)

    # Encode input sequences [a, op_token, b, equals_token]
    op_token = p      # ID for the operation
    eq_token = p + 1  # ID for '='
    seqs = np.stack([
        pairs[:, 0],                     # a
        np.full(len(pairs), op_token),   # op
        pairs[:, 1],                     # b
        np.full(len(pairs), eq_token)    # '='
    ], axis=1)

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(seqs))
    n_train = int(train_fraction * len(seqs))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = seqs[train_idx], results[train_idx]
    X_test,  y_test  = seqs[test_idx],  results[test_idx]

    # Convert to JAX arrays (int32)
    X_train = jnp.array(X_train, dtype=jnp.int32)
    y_train = jnp.array(y_train, dtype=jnp.int32)
    X_test  = jnp.array(X_test,  dtype=jnp.int32)
    y_test  = jnp.array(y_test,  dtype=jnp.int32)

    return X_train, y_train, X_test, y_test

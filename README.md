# Jax and Flax Grokking Transformer Implementation

This is a Jax and Flax implementation of the grokking phenomenon, where
neural networks exhibit a phase transition in generalization
performance during training.

**NOTE: this repo is a Jax/Flax port of the Grokking Modular Arithmetic - written in MLX by [Jason Stock](https://github.com/stockeh) - available on [https://github.com/stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)**

## Default Usage 

```bash
python main.py 
```

## Running Tests

The repository includes comprehensive unit and integration tests. To run all tests:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run specific test files
python -m pytest test_data.py -v
python -m pytest test_models.py -v
python -m pytest test_main.py -v
python -m pytest test_integration.py -v

# Run tests with coverage (optional)
pip install pytest-cov
python -m pytest --cov=. --cov-report=html
```

### Test Structure

- **`test_data.py`**: Unit tests for data generation (`data.py`)
  - Tests modular arithmetic correctness
  - Tests different operations (*, /, +, -)
  - Tests data formatting and train/test splits
  - Tests edge cases and error handling

- **`test_models.py`**: Unit tests for model components (`models.py`)
  - Tests RMSNorm layer normalization
  - Tests RoPE (Rotary Position Embeddings)
  - Tests MultiHeadSelfAttention with causal masking
  - Tests FeedForward layers with SiLU activation
  - Tests complete Transformer model

- **`test_main.py`**: Unit tests for training functions (`main.py`)
  - Tests loss computation and evaluation functions
  - Tests training step and gradient computation
  - Tests optimizer and learning rate scheduling
  - Tests model initialization and reproducibility

- **`test_integration.py`**: End-to-end integration tests
  - Tests complete training pipeline
  - Tests different model configurations
  - Tests batch size handling and memory efficiency
  - Tests gradient flow and model state consistency

## Overview

The implementation includes:
- Transformer-based architecture with RMSNorm and RoPE
- Customizable model parameters (depth, dimensions, heads)
- Learning rate warmup scheduler
- Training progress visualization
- Comprehensive test suite with 45+ test cases

## Architecture

The model uses:
- Transformer architecture with causal attention
- RMSNorm for layer normalization
- Rotary Position Embeddings (RoPE)
- AdamW optimizer with weight decay
- Learning rate warmup schedule

## Requirements
- jax[cpu]        # For CPU-based JAX; use jax for GPU/TPU as needed
- jaxlib
- flax
- optax
- numpy
- matplotlib
- tqdm
- pytest          # For testing

## Development

### Testing Philosophy

The test suite follows these principles:
- **Unit tests** validate individual components in isolation
- **Integration tests** verify the complete system works end-to-end
- **Reproducibility tests** ensure deterministic behavior with fixed seeds
- **Edge case testing** validates handling of boundary conditions
- **Performance tests** verify memory efficiency and gradient flow

### Adding New Tests

When adding new functionality, please include:
1. Unit tests for new functions/classes
2. Integration tests if the change affects the training pipeline
3. Edge case tests for parameter validation
4. Reproducibility tests for stochastic components

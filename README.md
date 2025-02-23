# Jax and Flax Grokking Transformer Implementation

This is a Jax and Flax implementation of the grokking phenomenon, where
neural networks exhibit a phase transition in generalization
performance during training.

**NOTE: this repo is a Jax/Flax port of the Grokking Modular Arithmetic - written in MLX by [Jason Stock](https://github.com/stockeh) - available on [https://github.com/stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)**

## Default Usage 

```bash
python main.py 
```

## Overview

The implementation includes:
- Transformer-based architecture with RMSNorm and RoPE
- Customizable model parameters (depth, dimensions, heads)
- Learning rate warmup scheduler
- Training progress visualization

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

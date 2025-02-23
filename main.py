# main.py
import time
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import data
import models

def main():
    # Hyperparameters
    p = 97
    operation = '/'
    train_fraction = 0.5

    depth = 2
    dim = 128
    heads = 1
    dropout = 0.2

    epochs = 150
    batch_size = 512

    learning_rate = 1e-3
    weight_decay = 1.0
    beta1 = 0.9
    beta2 = 0.98
    warmup_steps = 10

    seed = 42

    # 1) Prepare data
    X_train, y_train, X_val, y_val = data.grokking_data(
        p, op=operation, train_fraction=train_fraction, seed=seed
    )
    seq_len = X_train.shape[1]  # should be 4
    n_tokens = p + 2            # 0..p-1, plus op and '='

    # 2) Create model
    model = models.Transformer(
        depth=depth, dim=dim, heads=heads,
        n_tokens=n_tokens, seq_len=seq_len,
        dropout=dropout, pool='cls'
    )

    # 3) Initialize parameters with a JAX PRNGKey
    rng = jax.random.PRNGKey(seed)
    # "init" requires some sample input
    init_batch = X_train[:1]  # a single example
    params = model.init(rng, init_batch, training=True)['params']

    # 4) Create LR schedule with warmup
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

    # 5) Build optimizer: AdamW
    optimizer = optax.adamw(
        learning_rate=schedule_fn,
        b1=beta1,
        b2=beta2,
        eps=1e-8,
        weight_decay=weight_decay
    )

    # 6) Use TrainState to hold parameters & optimizer state
    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # 7) Define loss function
    def loss_fn(params, X, y, rng_key):
        logits = model.apply({'params': params}, X, training=True, rngs={'dropout': rng_key})
        one_hot = jax.nn.one_hot(y, n_tokens)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    # 8) Training step (JIT-compiled)
    @jax.jit
    def train_step(state, X, y, rng_key):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, X, y, rng_key)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(params=new_params, opt_state=new_opt_state)
        return new_state, loss

    # 9) Evaluation function
    @jax.jit
    def eval_step(params, X, y):
        logits = model.apply({'params': params}, X, training=False)
        one_hot = jax.nn.one_hot(y, n_tokens)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((preds == y).astype(jnp.float32))
        return loss, acc

    # 10) Training loop
    num_train = X_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))

    print(f"Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        perm = np.random.permutation(num_train)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        epoch_acc_count = 0
        for i in range(num_batches):
            batch_X = X_train[i * batch_size : (i+1) * batch_size]
            batch_y = y_train[i * batch_size : (i+1) * batch_size]

            rng, dropout_key = jax.random.split(rng)
            state, loss_val = train_step(state, batch_X, batch_y, dropout_key)
            epoch_loss += float(loss_val) * batch_X.shape[0]

        # Evaluate on training set for accuracy (optional, or do partial)
        # We'll skip a second forward pass on training set here. 
        train_loss = epoch_loss / num_train

        # Validation set
        val_loss, val_acc = eval_step(state.params, X_val, y_val)
        val_loss = float(val_loss)
        val_acc  = float(val_acc)

        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

    print("Training complete.")


if __name__ == "__main__":
    main()

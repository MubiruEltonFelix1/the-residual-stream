import jax
import jax.numpy as jnp
import numpy as np

"""
03_TRAINING_LOOP — JAX Implementation
Principles:
1. No libraries (no Optax/Flax for now) to keep gradients visible.
2. Numerical stability via the log-sum-exp trick.
3. Functional state management (params in, params out).
"""

def init_params(layers):
    """Initialize weights and biases for a simple MLP."""
    key = jax.random.PRNGKey(42)
    params = []
    for i in range(len(layers) - 1):
        key, subkey = jax.random.split(key)
        # Xavier/Glorot initialization for stability
        w_scale = jnp.sqrt(2.0 / (layers[i] + layers[i+1]))
        weight = jax.random.normal(subkey, (layers[i], layers[i+1])) * w_scale
        bias = jnp.zeros(layers[i+1])
        params.append((weight, bias))
    return params

def forward(params, x):
    """Simple MLP forward pass with ReLU activations."""
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    
    # Final layer (logits)
    w_last, b_last = params[-1]
    logits = jnp.dot(activations, w_last) + b_last
    return logits

def stable_cross_entropy(logits, labels):
    """
    Implements Cross-Entropy loss from scratch.
    Uses the log-sum-exp trick to prevent numerical overflow when computing softmax.
    """
    # log_softmax(x) = x - log(sum(exp(x)))
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits - jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=-1, keepdims=True))
    
    # NLL Loss: -mean(log_probs[correct_labels])
    n_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, n_classes)
    loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
    return loss

def loss_fn(params, x, y):
    logits = forward(params, x)
    return stable_cross_entropy(logits, y)

@jax.jit
def update(params, x, y, lr=0.01):
    """A single gradient descent step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    
    # Manual parameter update (Standard SGD)
    new_params = [
        (w - lr * dw, b - lr * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    return new_params, loss

# =====================================================================
# TINY TOY DATASET (For correctness testing)
# =====================================================================
if __name__ == "__main__":
    # 4 samples, 2 features, 2 classes (XOR-ish problem)
    X_train = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    Y_train = jnp.array([0, 1, 1, 0], dtype=jnp.int32)

    # Define architecture: [Input: 2, Hidden: 8, Output: 2]
    layers = [2, 8, 2]
    params = init_params(layers)

    print(f"Starting Training Loop...")
    for epoch in range(500):
        params, loss = update(params, X_train, Y_train, lr=0.1)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

    # Final Verification
    final_logits = forward(params, X_train)
    predictions = jnp.argmax(final_logits, axis=-1)
    print("-" * 30)
    print(f"Target Labels: {Y_train}")
    print(f"Predictions:   {predictions}")
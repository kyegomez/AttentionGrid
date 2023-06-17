# Documentation

**blockwise_compute_attn** Function:

The `blockwise_compute_attn` function is an important part of the `BlockwiseParallelJax` class and is used to compute the attention mechanism of the model in a blockwise manner.

Parameters:
* `query`, `key`, `value`: These parameters are the main inputs for the attention computation, representing queries, keys, and values, respectively.
* `bias`: Optional parameter used to add a bias to the attention scores before softmax.
* `deterministic`: A boolean flag used to decide whether or not to apply dropout.
* `dropout_rng`: The random number generator for dropout.
* `attn_pdrop`: The probability of dropout for attention.
* `causal_mask`: A boolean flag for whether or not to use a causal attention mask.
* `query_chunk_size`, `key_chunk_size`: The size of each query and key chunk, respectively.
* `dtype`: The data type of the computation. It's default is `jnp.float32`.
* `policy`: This parameter defines the policy for gradient checkpointing.
* `precision`: This parameter is used to set the level of precision for the computation. The default value is `lax.Precision.HIGHEST`.
* `prevent_cse`: A boolean flag used to prevent common subexpression elimination.

**blockwise_compute_ffn** Function:

The `blockwise_compute_ffn` function is used to compute the feed-forward network of the model in a blockwise manner.

Parameters:
* `cell`: The cell in the network to which the function is applied.
* `inputs`: The input data for the feed-forward network.
* `chunk_size`: The size of each chunk for the blockwise computation.
* `deterministic`: A boolean flag used to decide whether or not to apply dropout.
* `policy`: This parameter defines the policy for gradient checkpointing.
* `prevent_cse`: A boolean flag used to prevent common subexpression elimination.

**Blockwise_LM_Head** Class:

The `Blockwise_LM_Head` class is a module that applies a linear transformation followed by a softmax function to produce a distribution over the vocabulary for each position in the input.

* `vocab_size`: The size of the vocabulary, which is also the size of the output dimension of the linear transformation.
* `chunk_size`: The size of each chunk for the blockwise computation.
* `policy`: This parameter defines the policy for gradient checkpointing.
* `dtype`: The data type of the computation. It's default is `jnp.float32`.
* `prevent_cse`: A boolean flag used to prevent common subexpression elimination.

**blockwise_cross_entropy** Function:

The `blockwise_cross_entropy` function calculates the cross-entropy loss for the model's predictions in a blockwise manner.

Parameters:
* `logits`: The model's output predictions.
* `tokens`: The true labels.
* `valid`: A mask that specifies the valid positions in the input.
* `chunk_size`: The size of each chunk for the blockwise computation.
* `policy`: This parameter defines the policy for gradient checkpointing.
* `prevent_cse`: A boolean flag used to prevent common subexpression elimination.

**BlockwiseParallelJax** Class:

```python
BlockwiseParallelJax(q_chunk_size, k_chunk_size, hidden_size, num_heads, rotary_dim, intermediate_size, layer_norm_epsilon=1e-5, activation_function="gelu", attn_pdrop=0.0, resid_pdrop=0.0, max_position_embeddings=1024, dtype=jnp.float32, causal=True, policy='nothing_saveable', prevent_cse=False, float32_logits=False)
```

**Parameters**

- `q_chunk_size` : Integer. Chunk size for the query in self-attention.
- `k_chunk_size` : Integer. Chunk size for the key in self-attention.
- `hidden_size` : Integer. Dimensionality of the hidden layer in the transformer.
- `num_heads` : Integer. Number of attention heads in the self-attention mechanism.
- `rotary_dim` : Integer or None. Number of dimensions to use for rotary positional encoding.
- `intermediate_size` : Integer. Size of the intermediate layer in the feed-forward network.
- `layer_norm_epsilon` : Float. Small constant to prevent division by zero in layer normalization. Default is `1e-5`.
- `activation_function` : String. Activation function to use in the feed-forward network. Default is `'gelu'`.
- `attn_pdrop` : Float. Dropout probability for the attention mechanism. Default is `0.0`.
- `resid_pdrop` : Float. Dropout probability for the residual connections. Default is `0.0`.
- `max_position_embeddings` : Integer. Maximum number of position embeddings to use. Default is `1024`.
- `dtype` : jnp.dtype. Data type to use for computation. Default is `jnp.float32`.
- `causal` : Boolean. Whether to use causal (auto-regressive) mode or not. Default is `True`.
- `policy` : String. Policy for checkpointing gradients. Default is `'nothing_saveable'`.
- `prevent_cse` : Boolean. Whether to prevent common subexpression elimination (CSE). Default is `False`.
- `float32_logits` : Boolean. Whether to use float32 for logits computation. Default is `False`.

**Methods**

The main method of the `BlockwiseParallelJax` class is the `forward` method, which performs the forward pass of the transformer block.

```python
forward(hidden_states, attention_mask, position_ids, deterministic=True, init_cache=False)
```

- `hidden_states` : jnp.ndarray. The input tensor to the transformer block. It should have shape `(batch_size, sequence_length, hidden_size)`.
- `attention_mask` : jnp.ndarray. The attention mask for the self-attention mechanism. It should have shape `(batch_size, 1, 1, sequence_length)`.
- `position_ids` : jnp.ndarray. The position ids for positional encoding. It should have shape `(1, sequence_length)`.
- `deterministic` : Boolean. Whether to use deterministic mode (no dropout) or not. Default is `True`.
- `init_cache` : Boolean. Whether to initialize the cache for fast decoding. Default is `False`.

This method returns the output tensor of the transformer block, which has the same shape as `hidden_states`.

**Example Usage**

The following example demonstrates how to use the `BlockwiseParallelJax` class.

```python
# Initialize

from jax import random
import jax.numpy as jnp
from agora_ai_transformer.models import BlockwiseParallelJax

# Initialize transformer block
block = BlockwiseParallelJax(
    q_chunk_size=64,
    k_chunk_size=64,
    hidden_size=768,
    num_heads=12,
    rotary_dim=64,
    intermediate_size=3072,
)

# Create a batch of input tensors
key = random.PRNGKey(0)
batch_size = 8
sequence_length = 128
hidden_states = random.normal(key, (batch_size, sequence_length, block.hidden_size))

# Create attention mask
attention_mask = jnp.ones((batch_size, 1, 1, sequence_length))

# Create position ids
position_ids = jnp.arange(sequence_length)[None, :]

# Forward pass
output = block.forward(hidden_states, attention_mask, position_ids)

print(output.shape)  # prints: (8, 128, 768)
```
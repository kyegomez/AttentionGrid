from AttentionGrid import BlockwiseParallelJax
import jax.numpy as jnp

# Initialize the class
bpjax = BlockwiseParallelJax(
    q_chunk_size=64, 
    k_chunk_size=64, 
    hidden_size=768, 
    num_heads=12, 
    rotary_dim=32, 
    intermediate_size=3072
)

# Suppose we have hidden_states, attention_mask, and position_ids as input data
hidden_states = jnp.random.rand(1, 100, 768)
attention_mask = jnp.random.rand(1, 1, 100, 100)
position_ids = jnp.arange(100).reshape(1, 100)

# You can now apply the attention mechanism to your input data
output = bpjax.forward(hidden_states, attention_mask, position_ids)

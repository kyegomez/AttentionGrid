from AttentionGrid import BlockwiseParallel
import torch

# Initialize the class
bp = BlockwiseParallel(
    hidden_size=768, 
    num_heads=12, 
    rotary_dim=32, 
    intermediate_size=3072
)

# Suppose we have hidden_states, attention_mask, and position_ids as input data
hidden_states = torch.rand(1, 100, 768)
position_ids = torch.arange(100).unsqueeze(0)

# You can now apply the attention mechanism to your input data
output, attn_weights = bp(hidden_states, position_ids)

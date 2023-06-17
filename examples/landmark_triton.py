import torch
from AttentionGrid import fused_landmark_attention


# Define some input data
q = torch.randn(64, 12, 64)
k = torch.randn(64, 12, 64)
v = torch.randn(64, 12, 64)
is_mem = torch.randn(64, 12, 64)

# You can now use the fused landmark attention function in your code
output = fused_landmark_attention(q, k, v, is_mem)

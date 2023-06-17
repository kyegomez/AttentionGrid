from AttentionGrid import sparse_attn

# Define some hyperparameters
BATCH = 64
H = 12
n_ctx = 100
D_HEAD = 64

# You can now use the sparse attention function in your code
output = sparse_attn(num_buckets_or_sparsity=128, n_ctx=n_ctx, mode='fwd')

## Landmark Attention

### Class `FusedLandmarkAttention`

This is a PyTorch `Function` class that encapsulates the forward and backward functions of the Fused Landmark Attention mechanism. 

#### `forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size)`

This function performs the forward pass of the Fused Landmark Attention.

##### Parameters:

- `ctx`: An object to which we can save variables for use in the backward pass. Provided by PyTorch's autograd system.
- `q`: The queries tensor. It is assumed to be contiguous, and its shape should be (batch, nheads, seqlen_q, d).
- `k`: The keys tensor. It is assumed to be contiguous, and its shape should match q's shape, i.e., (batch, nheads, seqlen_k, d).
- `v`: The values tensor. It is assumed to be contiguous, and its shape should match q's and k's shapes, i.e., (batch, nheads, seqlen_k, d).
- `n_prefix_q`: The number of prefixes in the queries.
- `sm_scale`: The scaling factor used in the softmax operation.
- `block_size`: The block size for performing block-wise operations.

##### Returns:

- `o`: The output tensor from the forward pass of the Fused Landmark Attention mechanism.

#### `backward(ctx, do)`

This function performs the backward pass of the Fused Landmark Attention, i.e., it calculates the gradients.

##### Parameters:

- `ctx`: An object from which we can retrieve variables saved in the forward pass. Provided by PyTorch's autograd system.
- `do`: The gradient of the loss with respect to the output of the forward function.

##### Returns:

- A tuple containing the gradients of the loss with respect to the inputs to the forward function, in the same order as they were provided in. If a certain input does not require gradient, its corresponding gradient will be `None`.

### Function `fused_landmark_attention(q, k, v, is_mem, sm_scale=None, block_size=64)`

This function is a convenient wrapper for the `FusedLandmarkAttention` class.

##### Parameters:

- `q`: The queries tensor.
- `k`: The keys tensor.
- `v`: The values tensor.
- `is_mem`: A boolean tensor indicating whether each key-value pair should be treated as memory. It should have the same length as the sequence length of the keys.
- `sm_scale`: The scaling factor used in the softmax operation. If `None`, it will be set to `1.0 / sqrt(d)`.
- `block_size`: The block size for performing block-wise operations.

##### Returns:

- The output tensor from the forward pass of the Fused Landmark Attention mechanism.

#### Example:

Here is a basic example of how to use `fused_landmark_attention` function.

```python
import torch
from your_module import fused_landmark_attention

# Initialize some tensors
batch = 8
nheads = 12
seqlen = 128
d = 64
q = torch.randn(batch, nheads, seqlen, d)
k = torch.randn(batch, nheads, seqlen, d)
v = torch.randn(batch, nheads, seqlen, d)
is_mem = torch.zeros(seqlen, dtype=torch.bool)

# Call the function
output = fused_landmark_attention(q, k, v, is_mem)

print(output.shape)  # prints: (8, 12, 128, 64)
```

This example first initializes some tensors to serve as the queries, keys

, and values. Then it calls the `fused_landmark_attention` function and prints the shape of the output tensor.

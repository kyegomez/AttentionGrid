# Agora
![Agora Banner](agora-banner.png)

AttentionGrid is brought to you by Agora, we're an all-new open source multi-modal AI research organization devoted to advancing Humanity.

[Join us Here to contribute to this project or recieve support!](https://discord.gg/qUtxnK2NMf)

# AttentionGrid: Unleashing Attention Power in AI Models 🚀

![AttentionGrid Image](attention-grid.png)

AttentionGrid is a cutting-edge framework designed to democratize the incorporation of advanced attention mechanisms into AI models. Powered by the latest developments in attention-based transformer models, AttentionGrid opens up the world of attention mechanisms to machine learning practitioners, researchers, and enthusiasts alike.  


## Getting Started: Installation 🚀

To blast off with AttentionGrid, install the package using pip:

```bash
pip install AttentionGrid
```

Implementing an attention mechanism or a transformer model with AttentionGrid is as easy as:

```python
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
```

## Spread the Word 📣

We encourage you to share AttentionGrid with your community! Here are quick share links for several social media platforms:

- [Share on Twitter](https://twitter.com/intent/tweet?text=Check%20out%20AttentionGrid!%20An%20innovative%20framework%20for%20attention-based%20transformer%20models.%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAttentionGrid&hashtags=AI,ML,OpenSource)
  
- [Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAttentionGrid&title=AttentionGrid%3A%20Unleashing%20Attention%20Power%20in%20AI%20Models&summary=Check%20out%20AttentionGrid!%20An%20innovative%20framework%20for%20attention-based%20transformer%20models.)

- [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAttentionGrid)

- [Share on Reddit](http://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAttentionGrid&title=AttentionGrid:%20Unleashing%20Attention%20Power%20in%20AI%20Models)

- [Share on WhatsApp](https://wa.me/?text=Check%20out%20AttentionGrid!%20An%20innovative%20framework%20for%20attention-based%20transformer%20models.%20https%3A%2F%2Fgithub.com%2Fkyegomez%2FAttentionGrid)

Thank you for supporting AttentionGrid and contributing to the democratization of AI! Together, we can push the boundaries of what's possible.

## Vision 👁️

In the vast landscape of AI, attention mechanisms have revolutionized our ability to create powerful models that can discern the subtleties in data, focusing on important aspects and improving overall performance. Our vision with AttentionGrid is to bridge the gap between these state-of-the-art mechanisms and their practical applications, providing a tool that makes these techniques accessible and easy to implement in diverse AI applications.

## Architecture 🏗️

AttentionGrid is designed with an intuitive and flexible architecture, partitioned into four primary components:

1. **Core** 💡: This is the bedrock of our framework, housing abstract classes that layout the basic structure for attention mechanisms and transformer models.

2. **Attentions** 🧠: The directory dedicated to various attention mechanisms. Each attention mechanism is implemented based on the blueprint provided in the Core.

3. **Transformers** 🤖: This is where transformer models come to life, each sculpted following the design defined in the Core.

4. **Utils** 🛠️: A toolbox filled with helper classes for essential tasks like model loading, data preprocessing, and more.

5. **Examples** 🎯: Demystifying the implementation with hands-on examples and usage scenarios.



## Key Features ✨

- **Modular Structure**: Mix and match different attention mechanisms with a variety of transformer models.

- **User Friendly**: Clear documentation and examples to help you get started quickly.

- **Open Source**: Open to contributions, AttentionGrid thrives on collective knowledge and shared progress.


For more detailed examples, please refer to the 'examples' folder in our repository.

## Contribution 🤝

We openly invite contributions to AttentionGrid! Whether you have a new feature suggestion, bug report, or want to add to our code, please feel free to open an issue or submit a pull request.

## License 📜

AttentionGrid is proudly open-source software, licensed under the APACHE License.

## Why AttentionGrid? 🎯

Attention mechanisms have transformed AI, enabling machines to 'focus' on significant parts of input data. With AttentionGrid, we aim to democratize access to these powerful tools. We believe that the future of AI lies in the power of attention, and through AttentionGrid, we hope to accelerate this journey. Explore our repository, join our cause, and let's navigate this exciting landscape together!

> "The details are not the details. They make the design." - Charles Eames



# Roadmap

* Integrate Flash Attention, and variants

* Integrate landmark attention

* Integrate blockwise parallel attention

* [Integrate dynamic sparse flash attention](https://github.com/epfml/dynamic-sparse-flash-attention)

* Integrate cross attention from imagebind

* Integrate COLT-5 Attention

* Integrate multi-query attention 

* Integrate wrappers from lucid rains x_transformers, decoder, attention, encoder, transformer wrapper




# Documentation

## Dynamic Sparse Attention


Agora's `dynamic_sparse_attention` function allows the flexibility of choosing between the hash-sparse implementation and the qk-sparse implementation. This function's objective is to dynamically direct the sparse attention mechanism based on the selected `sparsity_mode`.

The function parameters are as follows:

- `q`: Query tensor of shape (BATCH, N_CTX_Q, H, D_HEAD)
- `k`: Key tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
- `v`: Value tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
- `q_idx` & `k_idx`: Represent either the bucket index if sparsity_mode is 'hash' or whether to keep a given head if sparsity_mode is 'qk'. The tensor shapes are (BATCH, N_CTX_Q, H) and (BATCH, N_CTX_KV, H) respectively.
- `sm_scale`: Normalization constant, 1/sqrt(D_HEAD) unless specified.
- `sparsity_mode`: 'hash' to select the hash-sparse implementation and 'qk' for the qk-sparse implementation.

The `sm_scale` is calculated by default if not provided, and if an unknown `sparsity_mode` is given, it throws a KeyError.

The function then checks the `sparsity_mode` and based on its value, it calls either `hash_sparse_attention` or `qk_sparse_attention`.

### Compact Function

The `compact` function builds a compact representation of the input tensor `x` using the information from `keep_tensor`.

The function parameters are:

- `x`: Input tensor to compact, with shape (BATCH, N_CTX, H, D_HEAD).
- `keep_tensor`: Float tensor of shape (BATCH, N_CTX, H) containing a 1 when the head is kept and 0 otherwise.

The function first calculates the `indices_per_head` which computes the number of non-killed elements per head. It sorts the `keep_tensor` in a descending order while preserving the order of equal elements (stable=True). It then gathers the elements of `x` based on the index tensor. The result is a compact representation of `x` along with the index tensor and the tensor representing the number of non-killed elements per head.

### Pad Index Function

The `pad_index` function pads the index tensor to comply with the kernel. It takes the following parameters:

- `index`: Original index tensor given by `compact`, with shape (BATCH, buffer_size, H). For each batch and timestep, it represents the head index it's originating from.
- `indices_per_head`: For each head, contains how many indices have not been dropped.

It creates a copy of the index tensor and creates a mask based on the size of `indices_per_head`. Then it modifies the indices in the copy that correspond to True in the mask to be equal to `pad_idx`.

### QK Sparse Attention Function

The `qk_sparse_attention` function is part of the dynamic sparse attention mechanism. It is used when `sparsity_mode` is set to 'qk'. This function implements the qk-sparse attention mechanism and requires that the `q_keep` and `k_keep` parameters are of type float.

It first builds compact representations of the query, key, and value tensors using the `compact` function. It then pads the index tensors using the `pad_index` function. The tensors are then transposed for compatibility with the kernel. Finally, the function calls the `qk_sparse_attention_kernel` function and scatters the resulting tensor back into the original dimension space.


### Hash Sparse Attention Function

The `hash_sparse_attention` function is part of the dynamic sparse attention mechanism. It is used when `sparsity_mode` is set to 'hash'. This function implements the hash-sparse attention mechanism.

The function takes the same input parameters as `qk_sparse_attention`. However, instead of `q_keep` and `k_keep` parameters, the `hash_sparse_attention` function requires `q_bucket_idx` and `k_bucket_idx` which represent bucket indices for queries and keys respectively.

The `hash_sparse_attention` function first sorts the query, key, and value tensors based on the bucket indices using the `sort_bucketed_attention` function. Then it builds compact representations of the sorted query, key, and value tensors using the `compact` function. It then pads the index tensors using the `pad_index` function. 

The tensors are then transposed for compatibility with the kernel. The function then calls the `hash_sparse_attention_kernel` function and scatters the resulting tensor back into the original dimension space.

### Sort Bucketed Attention Function

The `sort_bucketed_attention` function is a helper function used in `hash_sparse_attention`. It sorts the input tensors based on the given bucket indices.

The function parameters are:

- `qkv`: Query, Key, Value tensors of shape (BATCH, N_CTX, H, D_HEAD)
- `qkv_bucket_idx`: Bucket indices for queries, keys, and values of shape (BATCH, N_CTX, H)

The function first sorts the `qkv_bucket_idx` tensor and gets the sorted indices. Then it sorts the `qkv` tensors using the sorted indices. It also expands `qkv_bucket_idx` to be the same shape as `qkv` for compatibility.

### QK Sparse Attention Kernel Function

The `qk_sparse_attention_kernel` function is a kernel function used in `qk_sparse_attention`. It calculates the weighted sum of values based on the softmax of the query and key product. 

The function parameters are:

- `q`: Query tensor of shape (BATCH, N_CTX_Q, H, D_HEAD)
- `k`: Key tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
- `v`: Value tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
- `sm_scale`: Normalization constant, 1/sqrt(D_HEAD) unless specified.

### Hash Sparse Attention Kernel Function

The `hash_sparse_attention_kernel` function is a kernel function used in `hash_sparse_attention`. It works similarly to `qk_sparse_attention_kernel` but handles bucketing for hash-sparse attention.

The function parameters are the same as those of `qk_sparse_attention_kernel`. However, `q`, `k`, and `v` have been sorted and compacted based on the bucket indices. 

The kernel computes the product of the query and key, scales it by `sm_scale`, applies softmax to get the weights, and then calculates the weighted sum of the values.

Please note that this is a general interpretation of the documentation, and understanding and modifying these functions in practice may require in-depth knowledge of sparse attention mechanisms and deep learning principles.





## Blockwise Parallel Documentation

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
from AttentionGrid import BlockwiseParallelJax

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
from AttentionGrid import fused_landmark_attention

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



## LongNet

```python
import torch
import torch.nn as nn
from AttentionGrid import DilatedAttention

# Replace this with your correct GPU device
device = "cuda:0"
dtype = torch.float16

# Create an instance of DilatedAttention
d_model = 512
num_heads = 8
dilation_rate = 2
segment_size = 64
dropout = 0.2  # Specify the dropout rate
attention = DilatedAttention(
    d_model=d_model,
    num_heads=num_heads,
    dilation_rate=dilation_rate,
    segment_size=segment_size,
    dropout=dropout,
).to(device, dtype=dtype)

# Create some dummy input data
batch_size = 16
seq_len = 128
input_dim = d_model
inputs = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)

# Forward pass
outputs = attention(inputs)

# Print the output shape
print(outputs.shape)  # Expected: [batch_size, seq_len, d_model]
```

In the example above, we create an instance of the `DilatedAttention` class with the specified hyperparameters. We then generate some dummy input data and pass it through the attention mechanism to obtain the outputs. Finally, we print the shape of the output tensor.

## DilatedAttention Documentation

The `DilatedAttention` class implements dilated attention, which expands the attentive field exponentially as the distance between tokens grows. It inherits from `torch.nn.Module` and can be used as a drop-in replacement for standard attention mechanisms in Transformer models.

### Parameters

- `d_model` (int): The dimensionality of the input and output embeddings.
- `num_heads` (int): The number of attention heads.
- `dilation_rate` (int): The dilation rate for sparsifying the input sequence.
- `segment_size` (int): The size of each segment after sparsification.
- `dropout` (float, optional): The dropout probability to apply to the attention output. Default: 0.0 (no dropout).

### Inputs

- `x` (Tensor): The input tensor of shape `(batch_size, seq_len, d_model)`.

### Outputs

- `output` (Tensor): The output tensor of shape `(batch_size, seq_len, d_model)`.

Please note that the input tensor should be on the correct device (e.g., GPU) and have the appropriate data type (`dtype`).

## Agora Documentation 

### Dynamic Sparse Attention Function

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
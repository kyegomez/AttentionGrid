import torch
from blockwise_attention_torch import BlockwiseParallel

#constants
MASK_VALUE = -1e10


batch_size= 8 
sequence_length = 8192
hidden_size = 256
num_heads = 8
rotary_dim = 64
intermediate_size = 512

#random tensor
input_tensor = torch.randn(batch_size, sequence_length, hidden_size)


#create position_ids
position_ids = torch.arange(0, sequence_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

#create an attention mask {optional}
attention_mask = torch.ones(batch_size, sequence_length, sequence_length)
attention_mask = (attention_mask - torch.eye(sequence_length)).unsqueeze(1).repeat(1, sequence_length, 1)
attention_mask = attention_mask.masked_fill(attention_mask==0, MASK_VALUE)


#init attention block
attention_block = BlockwiseParallel(hidden_size, num_heads, rotary_dim, intermediate_size)

#pass the input tensor through the attention block
output, attn_weights = attention_block(input_tensor, position_ids, attention_mask)

#print the output tensor and attention weights
print(f"Output tensor shape: {output.shape}")

print("Attention weights shape: {attn_weights.shape}")
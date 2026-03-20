import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        bs, num_tokens, d_in = x.shape
        
        # Projecting the q, k, v 
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshaping them to [batch, n_heads, context_len, head_dim]
        queries = queries.view(bs, self.num_heads, num_tokens, self.head_dim)
        keys = keys.view(bs, self.num_heads, num_tokens, self.head_dim)
        values = values.view(bs, self.num_heads, num_tokens, self.head_dim)
        
        # Calculating attention scores
        attn_scores = queries @ keys.transpose(2,3)
        
        # truncating the mask to the size of current sequence
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        # Using the causal mask on the attention scores
        masked_attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        
        # Applying softmax on attention scores to produce attention weights
        masked_attn_weights = torch.softmax(
            masked_attn_scores / keys.shape[-1] ** 0.5,
            dim = -1
        )
        masked_attn_weights = self.dropout(masked_attn_weights)
        
        # Projecting the attention score on the value vector to produce context vector
        context_vec = masked_attn_weights @ values
        context_vec.transpose_(1,2)
        context_vec = context_vec.contiguous().view(bs, num_tokens, self.d_out)
        
        
        # Project the output using out_proj
        context_vec = self.out_proj(context_vec)
        
        return context_vec
        
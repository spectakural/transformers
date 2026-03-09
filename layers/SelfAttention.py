import torch.nn as nn
import torch

class SelfAttentionV3(nn.Module):
    def __init__(self, d_in, d_out, context_length, kqv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.context_length = context_length
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim = -1
        )
        context_len = attn_scores.shape[0]
        mask = torch.tril(torch.ones(context_len, context_len))
        
        masked_attn_weights = mask * attn_weights
        row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
        masked_norm_attn_weights = masked_attn_weights / row_sums
        
        context_vec = masked_norm_attn_weights @ values
        
        return context_vec
        
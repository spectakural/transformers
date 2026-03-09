import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=True
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(-2, -1)
        masked_attn_scores = torch.masked_fill(attn_scores, self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        masked_attn_weights = torch.softmax(
            masked_attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        
        context_vec = masked_attn_weights @ values
        
        return context_vec
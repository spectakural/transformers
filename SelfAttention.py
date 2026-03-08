import torch.nn as nn
import torch

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # Omega
        attn_weights = torch.softmax(
            (attn_scores / self.d_out ** 0.5), 
            dim=-1
        )
        context_vec = attn_weights @ values
        
        return context_vec
        
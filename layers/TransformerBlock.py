import torch
import torch.nn as nn

from transformers.setting import GPTConfig
from transformers.layers import (
    FeedForwardLayer,
    MultiheadAttention,
    LayerNorm,
    GELU
)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg.emb_dim)
        self.mha = MultiheadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            num_heads=cfg.n_heads,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.layer_norm2 = LayerNorm(cfg.emb_dim)
        self.ffn = FeedForwardLayer(cfg)
        
    def forward(self, x):
        y = self.layer_norm1(x)
        y = self.mha(y)
        y = self.dropout(y)
        
        x = x + y
        
        y = self.layer_norm2(x)
        y = self.ffn(y)
        y = self.dropout(y)
        
        x = x + y
        return x
import torch
import torch.nn as nn
from setting import GPTConfig
from .GELU import GELU

class FeedForwardLayer(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.ffn_depth_factor * cfg.emb_dim),
            GELU(),
            nn.Linear(cfg.ffn_depth_factor * cfg.emb_dim, cfg.emb_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
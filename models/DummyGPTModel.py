import torch
import torch.nn as nn

from setting import GPTConfig

class DummyGPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tokenizer = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(
            * [DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = DummyLayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        
    def forward(self, in_idx):
        bs, seq_len = in_idx.shape
        token_embeddings = self.tokenizer(in_idx)
        pos_embeddings = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x
    
        
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x
from pydantic import BaseModel

class GPTConfig(BaseModel):
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 22
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
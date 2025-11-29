import math
import torch 
from torch import nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        
    def forward(self, token):
        """
        return the multiplication of embeddings with sqrt of d_model. As it is mentioned in the AIAYN paper.
        > In the embedding layers, we multiply those weights by V(d_model)
        """
        return self.embeddings(token) * math.sqrt(self.d_model) 
        
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super.__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # PE matrix of shape [seq_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        
        # Numerator (positions) of shape [seq_len]
        positions = torch.arange(start=0, end=seq_len, step=1, dtype=torch.float)
        # denominator of shape [d_model/2] as for odd & even positions, it uses same denominator
        # here we use the mathematical identity -> a^(b) = exp(b * log(a)) | that is why it looks different than the one in AIAYN
        denominator = torch.exp(torch.arange(start=0, end=d_model, step=2) / d_model * -math.log(10000.0))
        
        # make the numerator and denominator broadcastable. Which means we want to have shapes like below:
        # positions: [seq_len, 1]
        # denominator: [1, d_model/2] 
        # So that when we multiply them, we get a matrix of shape [seq_len, d_model/2]
        positions = positions.unsqueeze(1)
        # denominator = denominator.unsqueeze(0) # this is not needed as PyTorch automatically broadcasts 1D tensors
        
        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        
        # add a batch dimension to the PE matrix
        pe = pe.unsqueeze(0)  # shape [1, seq_len, d_model]
        
        # register pe as a buffer so that it is not considered a model parameter, but is still saved and moved to GPU with the model
        # if you dont do this, pe will not be saved in model.state_dict() and will not be moved to GPU with model.to(device)
        # bc only parameters and registered buffers are moved to GPU with model.to(device) and while saving/loading state_dict
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor):
        return x + (self.pe[:, :x.size(1), :]).requires_grad_(False)  # add positional encoding to input embeddings
        # whether to apply dropout?

class LayerNormalization(nn.Module):
    """
    Implements the Layer Normalization Paper (https://arxiv.org/pdf/1607.06450)
    formula 
        xj_hat = (xj - mean_j)/sqrt(std_j^2 + eps)
        
    """
    def __init__(self, eps: float = 10e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.register_buffer("eps", eps)
        
    def forward(self, x):
        mean = torch.mean(input=x, dim=-1, keepdim=True)
        std = torch.std(input=x, dim=-1, keepdim=True)
        return (x - mean)/torch.sqrt(std + self.eps)*self.alpha + self.bias
        # no need of sqrt??
    
    
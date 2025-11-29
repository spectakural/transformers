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
        
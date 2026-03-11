import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        product = 1 + torch.tanh(
            torch.sqrt(2.0 / torch.pi) * (x + torch.pow(x, 3) + 0.044715)
        )
        return 0.5 * x * product
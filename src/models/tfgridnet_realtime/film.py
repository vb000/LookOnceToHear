import torch.nn as nn
import torch

class FilmLayer(nn.Module):
    def __init__(self, D, C, nF):
        super().__init__()
        self.D = D
        self.C = C
        self.nF = nF

        self.weight = nn.Conv1d(self.D, self.C * nF, 1)
        self.bias = nn.Conv1d(self.D, self.C * nF, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, F, T)
        embedding: (B, D, F)
        """
        B, D, _F, T = x.shape
        w = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)
        b = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)

        return x * w + b
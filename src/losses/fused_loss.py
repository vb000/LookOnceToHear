import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR
import torch


class FusedLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdrloss = SingleSrcNegSDR('sisdr')
        self.snrloss = SingleSrcNegSDR('snr')
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        sisdr_losses = self.sisdrloss(est, gt)
        snr_losses = self.snrloss(est, gt)
        batch_losses = sisdr_losses + snr_losses
        return torch.mean(batch_losses)

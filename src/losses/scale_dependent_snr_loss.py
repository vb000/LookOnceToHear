import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR
import torch


class SDSDR_SNR_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sdsdrloss = SingleSrcNegSDR('sdsdr')
        self.snrloss = SingleSrcNegSDR('snr')
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        sdsdr_losses = self.sdsdrloss(est, gt)
        snr_losses = self.snrloss(est, gt)
        batch_losses = torch.maximum(sdsdr_losses, snr_losses)
        return torch.mean(batch_losses)

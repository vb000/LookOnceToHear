import torch.nn as nn
from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses.pit_wrapper import PITLossWrapper
import torch


class SISDR_with_PIT_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdrloss = PITLossWrapper(PairwiseNegSDR("sisdr"))
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, est1, est2, gt1, gt2):
        """
        est1: (B, C, T)
        est2: (B, C, T)
        gt1: (B, C, T)
        gt2: (B, C, T)
        """
        est = torch.cat([est1.unsqueeze(2), est2.unsqueeze(2)], dim=2)
        gt = torch.cat([gt1.unsqueeze(2), gt2.unsqueeze(2)], dim=2)

        B, C, S, T = est.shape
        est = est.reshape(B * C, S, T)
        gt = gt.reshape(B * C, S, T)

        sisdrloss, reordered = self.sisdrloss(est, gt, return_est=True)
        
        reordered = reordered.reshape(B, C, S, T) # This contains reordered estimates
        
        return sisdrloss, reordered[:, :, 0], reordered[:, :, 1]

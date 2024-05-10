import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR
import torch


class SISDR_with_sum_loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdrloss = SingleSrcNegSDR('sisdr')
        self.l1 = nn.L1Loss()
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, noise_estimate: torch.Tensor, mixture: torch.Tensor):
        sdsdr_losses = self.sisdrloss(est, gt)
        
        B, C, T = noise_estimate.shape
        noise_estimate = noise_estimate.reshape(-1, T)
        mixture = mixture.reshape(-1, T)
        mix_est = noise_estimate + est

        loss = torch.mean(sdsdr_losses) + self.l1(mix_est, mixture)

        return loss

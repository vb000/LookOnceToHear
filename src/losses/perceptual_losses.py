import torch.nn as nn
from cdpam import CDPAM
import auraloss
import torch


class CDPAMLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.cdpam = CDPAM()
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        """
        est, gt: (B*C, T)
        """
        return torch.mean( self.cdpam.forward(gt, est) )

class MultiResolutionMelSpecLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.stftloss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [1024,2048,8192],hop_sizes=[256,512,2048], win_lengths=[1024,2048,8192], scale='mel', n_bins=128, sample_rate=16000, perceptual_weighting=True)
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        """
        est, gt: (B*C, T)
        """
        return self.stftloss(est.unsqueeze(1), gt.unsqueeze(1))

class L1_Mel(nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.stftloss = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes = [64, 128, 256, 1024],hop_sizes=[128, 256, 512, 2048], win_lengths=[64,128,256, 1024], n_bins=None, sample_rate=16000, device=device)
        # self.stftloss = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes = [64, 128, 256, 1024],hop_sizes=[128, 256, 512, 2048], win_lengths=[64,128,256, 1024], scale='mel', n_bins=None, sample_rate=16000, perceptual_weighting=True, device=device)
        self.l1loss = nn.L1Loss()
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        """
        est, gt: (B*C, T)
        """
        BC, T = est.shape
        l1 = self.stftloss(est.reshape(-1, 2, T), gt.reshape(-1, 2, T))
        l2 = self.l1loss(est, gt).mean()
        return l1 + l2

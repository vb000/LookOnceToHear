import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.tfgridnet_separator import TFGridNet
from .stft_decoder import STFTDecoder

class TFGridNet(TFGridNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_fft = kwargs['n_fft']
        stride = kwargs['stride']
        self.dec = STFTDecoder(n_fft, n_fft, stride, window="hann")
        self.n_fft = n_fft

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization

        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

class Net(nn.Module):
    def __init__(self, num_ch, n_fft, stride, num_blocks):
        super().__init__()
        self.net = TFGridNet(
            input_dim=None, n_fft=n_fft, stride=stride, n_imics=num_ch, n_srcs=2,
            lstm_hidden_units=64, n_layers=num_blocks)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, M, N] -> [B, N, M]
        ilens = torch.tensor([_.shape[0] for _ in x])
        x, _, _ = self.net(x, ilens)
        x = torch.stack(x, dim=1) # [B, 2, T]
        return x

class EmbedTFGridNet(TFGridNet):
    def __init__(self, embed_dim, num_ch, n_fft, stride, num_blocks):
        super().__init__(
            input_dim=None, n_fft=n_fft, stride=stride, n_imics=num_ch, n_srcs=1,
            lstm_hidden_units=64, n_layers=num_blocks, emb_dim=64)
        self.emb_dim = 64
        self.n_freqs = n_fft // 2 + 1
        self.embed_proj = nn.Sequential(
            nn.Linear(self.n_freqs * self.emb_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input):
        input = input.transpose(1, 2) # [B, M, N] -> [B, N, M]
        ilens = torch.tensor([_.shape[0] for _ in input])

        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = batch.permute(0, 2, 1, 3)  # [B, T, -1, F]
        batch = batch.reshape([n_batch, n_frames, -1])  # [B, T, -1]
        batch = self.embed_proj(batch) # [B, T, embed_dim]
        batch = batch.mean(dim=1) # [B, embed_dim]

        return batch

import torch
import torch.nn as nn

from .tfgridnet_causal import TFGridNet
import torch.nn.functional as F


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Net(nn.Module):
    def __init__(self, stft_chunk_size=160, stft_pad_size = 120, embed_dim=256,
                 num_ch=2, D=64, B=6, I=1, J=1, L=0, H=128,
                 use_attn=False, lookahead=True, local_atten_len=100,
                 chunk_causal=False, num_src = 2):
        super(Net, self).__init__()
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_chunk_size + stft_pad_size

        # TF-GridNet        
        self.tfgridnet = TFGridNet(None,
                                   n_srcs=num_src,
                                   n_fft=self.nfft,
                                   spk_emb_dim=embed_dim,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=B,
                                   n_imics=num_ch,
                                   attn_n_head=L,
                                   use_attn = use_attn,
                                   lstm_hidden_units=H,
                                   local_atten_len=local_atten_len,
                                   chunk_causal = chunk_causal)

    def init_buffers(self, batch_size, device):
        return self.tfgridnet.init_buffers(batch_size, device)

    def predict(self, x, embed, input_state, pad=True):
        mod = 0
        if pad:
            pad_size = (0, self.stft_pad_size) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)
        
        x, next_state = self.tfgridnet(x, embed, input_state)
        x = x[..., : -self.stft_pad_size]
        
        if mod != 0:
            x = x[:, :, :-mod]

        return x, next_state

    def forward(self, x, embeds, input_state = None, pad=True):
        embeds = embeds[:, 0] # [B, E]
        
        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)
        
        x, next_state = self.predict(x, embeds, input_state, pad)

        return x

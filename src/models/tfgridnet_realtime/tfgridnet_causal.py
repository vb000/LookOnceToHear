import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.get_layer_from_string import get_layer

from asteroid_filterbanks import make_enc_dec

from .film import FilmLayer

# More efficient TFGridNet:
# 1- Causal/Streaming attention, compute attention on first chunk only by setting Q to be the last chunk
# 2- Pad separate context buffers for Q, K, V after linear layer
# 3- Scale channels by num_heads and then reshape output to avoid for loop


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x

class TFGridNet(AbsSeparator):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        input_dim,
        spk_emb_dim=256,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=1,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        ref_channel=-1,
        use_attn=True,
        chunk_causal=True,
        local_atten_len=100
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.ref_channel = ref_channel
        self.emb_dim = emb_dim
        self.eps = eps
        
        self.chunk_size = stride
        
        self.istft_pad = n_fft - stride
        
        # ISTFT overlap-add will affect this many chunks in the future
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad
        
        self.enc, self.dec = make_enc_dec('stft',
                                n_filters = n_fft,
                                kernel_size = n_fft,
                                stride=stride,
                                window_type=window)

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, 3), (0, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            # nn.GroupNorm(1, emb_dim, eps=eps),
            # LayerNormPermuted(emb_dim)
        )

        self.embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    use_attn=use_attn,
                    chunk_causal=chunk_causal,
                    local_atten_len=local_atten_len
                )
            )

        self.embed_to_feats_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, emb_dim * n_freqs),
            nn.LayerNorm(emb_dim * n_freqs)
        )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))
    
    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, self.n_imics*2, self.t_ksize - 1, self.n_freqs,
                              device=device)
        deconv_buf = torch.zeros(batch_size, self.emb_dim, self.t_ksize - 1, self.n_freqs,
                              device=device)
        istft_buf = torch.zeros(batch_size, self.n_srcs, self.n_freqs * 2, self.istft_lookback,
                                device=device)

        gridnet_buffers = {}
        for i in range(len(self.blocks)):
            gridnet_buffers[f'buf{i}'] = self.blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    istft_buf=istft_buf, gridnet_bufs=gridnet_buffers)
    
    def forward(
        self,
        input: torch.Tensor,
        spk_embedding: torch.Tensor,
        input_state
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns: # MODIFIED: WILL RETURN [B, M, N] !!!
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """

        # print("input", input.shape)
        n_samples = input.shape[-1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input.unsqueeze(1)  # [B, M, N]

        if input_state is None:
            input_state = self.init_buffers(input.shape[0], input.device)
        
        conv_buf = input_state['conv_buf']
        deconv_buf = input_state['deconv_buf']
        gridnet_buf = input_state['gridnet_bufs']
        istft_buf = input_state['istft_buf']
        # mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        # input = input / mix_std_  # RMS normalization
        
        # t0 = time.time()
        # batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch = self.enc(input) # [B, M, nfft + 2, T]
        
        batch = torch.cat((batch[..., :self.n_freqs, :], batch[..., self.n_freqs:, :]), dim=1)  # [B, 2*M, F, T]
        batch = batch.transpose(2, 3) # [B, M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape # B, 2M, T, F
        
        # print(conv_buf.shape, batch.shape)

        ## pad init buffer 
        batch = batch
        batch = torch.cat(( conv_buf, batch), dim=2)
        conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.conv(batch)  # [B, -1, T, F]
        # t1 = time.time()

        # print("batch conv", batch.shape)
        # ts = [0, 0, 0, 0, 0, 0 ]
        embed = self.embed_to_feats_proj(spk_embedding) # [B, C * F]
        embed = embed.reshape([n_batch, self.emb_dim, n_freqs]).unsqueeze(2) # [B, C, 1, F]
        for ii in range(self.n_layers):
            if ii == 1:
                batch = batch * embed
            batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])  # [B, -1, T, F]
            # ts = [ts[i] + deltas[i] for i in range(len(ts))]
            
        # t2 = time.time()
        batch = torch.cat(( deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]batch ] 
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, 2, n_frames, n_freqs]
        
        
        batch = batch.transpose(3, 4) # (B, n_srcs, 2, n_fft//2 + 1, T)
        
        # Concat real and imaginary parts
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2) # (B, n_srcs, nfft + 2, T)
        
        # Cat istft from previous chunks
        batch = torch.cat([istft_buf, batch], dim=3)
        istft_buf = batch[..., -self.istft_lookback:]

        batch = self.dec(batch) # [B, n_srcs, n_srcs, -1]
        batch = batch[..., self.istft_lookback * self.chunk_size:]
        
        # Do not pad here. This will be taken care of outside

        # batch = batch * mix_std_  # reverse the RMS normalization
        input_state['conv_buf'] = conv_buf
        input_state['deconv_buf'] = deconv_buf
        input_state['gridnet_bufs'] = gridnet_buf
        input_state['istft_buf'] = istft_buf

        return batch, input_state

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        local_atten_len= 100,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        use_attn=True,
        chunk_causal = True
    ):
        super().__init__()
        bidirectional = False # Causal
        self.use_attn = use_attn
        self.local_atten_len = local_atten_len
        self.E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
        self.n_head = n_head
        self.V_dim = emb_dim // n_head
        self.chunk_causal = chunk_causal
        self.H = hidden_channels
        # if chunk_causal:
        #     self.unfold2 = nn.Unfold(kernel_size=(local_atten_len, 1), stride=1)

        in_channels = emb_dim
        self.in_channels = in_channels
        self.n_freqs = n_freqs

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        ## intra RNN can be optimized by conv or linear because the frequence length are not very large
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.Linear(
            hidden_channels*2, emb_dim,
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM( 
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels*(bidirectional + 1), emb_dim
        )
        
        if self.use_attn:
            E = self.E
            assert emb_dim % n_head == 0
            self.add_module(
                "attn_conv_Q",
                nn.Sequential(
                    nn.Linear(emb_dim, E * n_head), # [B, T, Q, HE]
                    get_layer(activation)(),
                    # [B, T, Q, H, E] -> [B, H, T, Q, E] ->  [B * H, T, Q * E]
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)), # (BH, T, Q * E)
                    LayerNormalization4DCF((n_freqs, E), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K",
                nn.Sequential(
                    nn.Linear(emb_dim, E * n_head),
                    get_layer(activation)(),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)),
                    LayerNormalization4DCF((n_freqs, E), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V",
                nn.Sequential(
                    nn.Linear(emb_dim, (emb_dim // n_head) * n_head),
                    get_layer(activation)(),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, (emb_dim // n_head))\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * (emb_dim // n_head))),
                    LayerNormalization4DCF((n_freqs, emb_dim // n_head), eps=eps),
                ),
            )
            self.add_module(
                "attn_concat_proj",
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    get_layer(activation)(),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                    LayerNormalization4DCF((n_freqs, emb_dim), eps=eps)
                ),
            )
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _init_buffers(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.in_channels, self.local_atten_len - 1, self.n_freqs),
            device=device)
    
    def init_buffers(self, batch_size, device):
        K_buf = torch.zeros((batch_size * self.n_head,
                             self.local_atten_len - 1,
                             self.E * self.n_freqs), device=device)
        V_buf = torch.zeros((batch_size * self.n_head,
                             self.local_atten_len - 1,
                             self.V_dim * self.n_freqs), device=device)
        
        c0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        
        h0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)

        return dict(K_buf=K_buf,
                    V_buf=V_buf,
                    c0=c0,
                    h0=h0)

    def _causal_unfold_chunk(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.
        # NOTE: THE COMMENTS HERE ARE UNRELIABLE

        Args:
            x: [B, C, F, T], L is total length of signal
            ctx_len: int
        Returns:
            [B, num_chunk,  (ctx_len + chunk_size), C]
        """

        # print(x.shape)
        x = x.transpose(1, 2)
        
        if x.shape[-1] == self.local_atten_len:
            return x
        
        # print('A', x.shape)
        x = x.unfold(2, self.local_atten_len, 1) # [B, num_chunk, CF, atten_len]
        
        B, CF, N, L = x.shape
        x = x.transpose(1, 2).reshape(B * N, CF, L)

        return x
        
    def get_lookahead_mask(self, seq_len, device):
        """Creates a binary mask for each sequence which maskes future frames.
        Arguments
        ---------
        padded_input: torch.Tensor
            Padded input tensor.
        Example
        -------
        >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
        >>> get_lookahead_mask(a)
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0.]])
        """
        if(seq_len <= self.local_atten_len):
            mask = (
                torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
            ).transpose(0, 1)
        else:
            mask1 = torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
            mask2 = torch.triu(torch.ones((seq_len, seq_len), device=device),  diagonal = self.local_atten_len) == 0
            mask = (
               mask1*mask2
            ).transpose(0, 1) 

        
        # mask = (
        #     mask.float()
        #     .masked_fill(mask == 0, float("-inf"))
        #     .masked_fill(mask == 1, float(0.0))
        # )
        return mask.detach().to(device)

    def forward(self, x, init_state = None):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        
        if init_state is None:
            init_state = self.init_buffers(batch.shape[0], Q.device)

        B, C, old_T, old_Q = x.shape
        T = old_T#math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = old_Q#math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        
        # intra RNN
        input_ = x.permute(0, 2, 3, 1) # [B, T, Q, C]
        
        intra_rnn = self.intra_norm(input_) # [B, T, Q, C]
        intra_rnn = intra_rnn.reshape(B * T, Q, C)  # [B * T, Q, C]
        
        self.intra_rnn.flatten_parameters()
        
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q, C]
        
        intra_rnn = intra_rnn.reshape(B, T, Q, C) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        # inter RNN
        input_ = intra_rnn # [B, T, Q, C]
        
        inter_rnn = self.inter_norm(intra_rnn)  # [B, T, Q, C]
        inter_rnn = (inter_rnn.transpose(1, 2).reshape(B * Q, T, C))  # [BQ, T, C]
        
        self.inter_rnn.flatten_parameters()
        
        h0 = init_state['h0']
        c0 = init_state['c0']

        inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [BQ, -1, H]
       
        init_state['h0'] = h0
        init_state['c0'] = c0
       
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T, C]
        
        inter_rnn = inter_rnn.view([B, Q, T, C])
        inter_rnn = inter_rnn.transpose(1, 2) # [B, T, Q, C]
        inter_rnn = inter_rnn + input_  # [B, T, Q, C]
        
        if self.chunk_causal:
            out = inter_rnn
            B0, T0, Q0, C0 = inter_rnn.shape
            batch = inter_rnn
            
            if self.use_attn:
                # attention
                Q = self["attn_conv_Q"](batch) # [B', T, Q * C]
                
                K = self["attn_conv_K"](batch) # [B', T, Q * C]
                
                V = self["attn_conv_V"](batch) # [B', T, Q * C]
                
                K_buf = init_state['K_buf']
                V_buf = init_state['V_buf']

                K = torch.cat([K_buf, K], dim = 1)
                start = K.shape[1] - (self.local_atten_len-1)
                init_state['K_buf'] = K[:, start:start+self.local_atten_len - 1]
                
                V = torch.cat([V_buf, V], dim = 1)
                start = V.shape[1] - (self.local_atten_len-1)
                init_state['V_buf'] = V[:, start:start+self.local_atten_len - 1]
                
                Q = Q.reshape(Q.shape[0]*Q.shape[1], 1, Q.shape[2]) # [B'T, 1, Q * C]
                
                K = self._causal_unfold_chunk(K)
                V = self._causal_unfold_chunk(V)
                
                emb_dim = Q.shape[-1]

                attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
                attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]

                V = torch.matmul(attn_mat, V.transpose(1, 2))  # [B', T, Q*C]
                V = V.reshape(-1, T0, V.shape[-1]) # [BH, T, Q * C]
                V = V.transpose(1, 2) # [B', Q * C, T]
                
                batch = V.reshape(B0, self.n_head, self.n_freqs, self.V_dim, T0) # [B, H, Q, C, T]
                batch = batch.transpose(2, 3) # [B, H, C, Q, T]
                batch = batch.reshape(B0, self.n_head * self.V_dim, self.n_freqs, T0) # [B, HC, Q, T]
                batch = batch.permute(0, 3, 2, 1) # [B, T, Q, HC]
                
                batch = self["attn_concat_proj"](batch) # [B, T, Q * C]
                batch = batch.reshape(batch.shape[0], batch.shape[1], out.shape[2], -1)  # [B, T, Q, C])

                # Add batch if attention is performed
                out = out + batch
                out = out.permute(0, 3, 1, 2)

        return out, init_state#, [t0 - t0_0, t1 - t0, t2 - t2_0, t3 - t2, t5 - t4, t7 - t6]


# Use native layernorm implementation
class LayerNormalization4D(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.norm = nn.LayerNorm(C, eps=eps)
        self.preserve_outdim = preserve_outdim

    def forward(self, x: torch.Tensor):
        """
        input: (*, C)
        """
        x = self.norm(x)
        return x
    
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        assert len(input_dimension) == 2
        Q, C = input_dimension
        super().__init__()
        self.norm = nn.LayerNorm((Q * C), eps=eps)

    def forward(self, x: torch.Tensor):
        """
        input: (B, T, Q * C)
        """
        x = self.norm(x)

        return x
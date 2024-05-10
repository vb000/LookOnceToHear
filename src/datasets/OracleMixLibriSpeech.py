"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
import logging
import warnings

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset
import scaper

from src.datasets.multi_ch_simulator import CIPICSimulator

# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")

class OracleDataset(Dataset):
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, dset, sr=None,
                 resample_rate=None, num_enroll=10, enroll_len=5) -> None:
        super().__init__()
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"

        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.hrtf_list = hrtf_list
        self.jams_dir = jams_dir
        self.embed_dir = embed_dir
        self.dset = dset
        self.sr = sr
        self.resample_rate = resample_rate
        self.num_enroll = num_enroll

        logging.info(f"Loading dataset: {dset} {sr=} {resample_rate=} ...")
        logging.info(f"  - Foreground directory: {fg_dir}")
        logging.info(f"  - Background directory: {bg_dir}")
        logging.info(f"  - Embedding directory: {embed_dir}")
        logging.info(f"  - JAMS directory: {jams_dir}")
        logging.info(f"  - HRTF directory: {hrtf_list}")

        self.samples = sorted(list(Path(self.jams_dir).glob('[0-9]*')))

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr
        self.enroll_len = enroll_len * self.sr

        self.multi_ch_simulator = CIPICSimulator(hrtf_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load Audio
        jamsfile = os.path.join(sample_dir, 'mixture.jams')
        _, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)

        # Squeeze to mono
        event_audio_list = [x.squeeze(1) for x in event_audio_list]

        # Simulate multi-channel audio
        multi_ch_seed = idx
        if self.dset == 'train':
            multi_ch_seed = random.randrange(1, 100000)
        multi_ch_events, multi_ch_noise = self.multi_ch_simulator.simulate(
            event_audio_list[1:], event_audio_list[0], self.sr, multi_ch_seed)

        # To torch
        multi_ch_events = [torch.from_numpy(x).float() for x in multi_ch_events]
        multi_ch_noise = torch.from_numpy(multi_ch_noise).float()

        # Normalize and compute mixture
        norm_factor = torch.abs(sum(multi_ch_events) + multi_ch_noise).max()
        for i in range(len(multi_ch_events)):
            multi_ch_events[i] /= norm_factor
        multi_ch_noise /= norm_factor
        mixture = sum(multi_ch_events) + multi_ch_noise

        # Select target index
        if self.dset == 'train':
            tgt_idx = random.randrange(len(multi_ch_events))
        else:
            _rng = random.Random(idx)
            tgt_idx = _rng.randrange(len(multi_ch_events))
 
        # Resample
        mixture = self.resampler(mixture)
        
        sources = torch.stack([*multi_ch_events, multi_ch_noise])
        sources = torch.stack([self.resampler(target) for target in sources])
        
        tgt_source = sources[tgt_idx].clone()
        sources[tgt_idx] = sources[0]
        sources[0] = tgt_source

        inputs = {
            'mixture': mixture,
            'sources': sources,
        }

        targets = {
            'target': tgt_source
        }

        return inputs, targets

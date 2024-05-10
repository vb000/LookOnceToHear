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


class MixLibriSpeechCIPICDatasetDM(Dataset):
    """
    LibriSpeech dataset with dynamic mixing. Only to be used for training set.
    """
    def __init__(self, fg_dir, bg_dir, embed_dir, hrtf_list, dset, sr=None,
                 resample_rate=None, num_enroll=10, enroll_len=5,
                 num_samples=100000,
                 num_events_min=2, num_events_max=3, duration=5.0,
                 event_duration_min=5.0, event_duration_max=5.0, 
                 ref_db=-25.0, snr_min=15.0, snr_max=25.0) -> None:
        super().__init__()
        assert dset in ['train'], \
            "Dynamic Mixing must only be used during training!"

        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.hrtf_list = hrtf_list
        self.embed_dir = embed_dir
        self.dset = dset
        self.sr = sr
        self.resample_rate = resample_rate
        self.num_enroll = num_enroll
        self.num_samples = num_samples

        # Initialize scaper object and store dynamic mixing parameters
        self.sc = scaper.Scaper(duration, self.fg_dir, self.bg_dir)
        
        self.sc.ref_db = ref_db
        self.sc.sr = self.sr
        self.num_events_min = num_events_min
        self.num_events_max = num_events_max
        self.event_duration_min = event_duration_min
        self.event_duration_max = event_duration_max
        self.snr_min = snr_min
        self.snr_max = snr_max

        logging.info(f"Loading dataset: {dset} {sr=} {resample_rate=} ...")
        logging.info(f"  - Foreground directory: {fg_dir}")
        logging.info(f"  - Background directory: {bg_dir}")
        logging.info(f"  - Embedding directory: {embed_dir}")
        logging.info(f"  - HRTF directory: {hrtf_list}")

        logging.info(f"USING DYNAMIC MIXING: {hrtf_list}")

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr
        self.enroll_len = enroll_len * self.sr

        self.multi_ch_simulator = CIPICSimulator(hrtf_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load Audio
        jams, ann_list, event_audio_list = self.choose_sources()

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
        tgt_idx = random.randrange(len(multi_ch_events))

        # Target event
        target = multi_ch_events[tgt_idx]
        tgt_id = ann_list[tgt_idx][-1]

        # Get embeddings
        embed_path = os.path.join(self.embed_dir, tgt_id + '.pt')
        embeds = torch.load(embed_path, map_location='cpu').items()
        if self.dset == 'train':
            embeds = random.sample(embeds, self.num_enroll)
        else:
            _rng = random.Random(idx)
            embeds = _rng.sample(embeds, self.num_enroll)
        embed_paths = [os.path.join(self.fg_dir, tgt_id, x[0]) for x in embeds]
        embeds = torch.cat([torch.from_numpy(x[1]) for x in embeds], dim=0)

        # Get enrollment audio and pad or crop to same length
        enrollments = [
            self.resampler(torchaudio.load(f)[0]) for f in embed_paths
        ]
        for enroll_idx, enroll in enumerate(enrollments):
            if enroll.shape[-1] < self.enroll_len:
                enrollments[enroll_idx] = torch.nn.functional.pad(
                    enroll, (0, self.enroll_len - enroll.shape[-1]))
            enrollments[enroll_idx] = enrollments[enroll_idx][..., :self.enroll_len]
        enrollments = torch.stack(enrollments)
 
        # Resample
        mixture = self.resampler(mixture)
        target = self.resampler(target)

        inputs = {
            'mixture': mixture,
            'embeds': embeds,
            'enrollments': enrollments,
            'embed_paths': embed_paths
        }

        targets = {
            'target': target
        }

        return inputs, targets

    def choose_sources(self):
        # Reset event specifications
        self.sc.reset_bg_event_spec()
        self.sc.reset_fg_event_spec()

        # add background
        self.sc.add_background(label=('choose', ['tr']), source_file=('choose', []),
                               source_time=('const', 0))

        # Add random number of foreground events
        n_events = np.random.randint(
            self.num_events_min, self.num_events_max + 1)
        for _ in range(n_events):
            self.sc.add_event(
                label=('choose', []),
                source_file=('choose', []),
                source_time=('const', 0.0),
                event_time=('uniform', 0.0, 1.0),
                event_duration=(
                    'uniform', self.event_duration_min,
                    self.event_duration_max),
                snr=('uniform', self.snr_min, self.snr_max),
                pitch_shift=None,
                time_stretch=None)
        
        _, jams, ann_list, event_audio_list = self.sc.generate(
                                                    allow_repeated_label=False,
                                                    allow_repeated_source=False,
                                                    reverb=None,
                                                    disable_sox_warnings=True,
                                                    no_audio=False,
                                                    save_isolated_events=False,
                                                    fix_clipping=True,
                                                    disable_instantiation_warnings=True)
        
        return jams, ann_list, event_audio_list

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
import pandas as pd

from src.datasets.motion_simulator import CIPICMotionSimulator2

# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")

class MixLibriSpeechCIPICMotionDataset(Dataset):
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, dset,
                 sr=None, resample_rate=None, num_enroll=10, enroll_len=5) -> None:
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
        self.simulation_sr = sr
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

        self.multi_ch_simulator = CIPICMotionSimulator2(hrtf_list, self.simulation_sr)

        # Get speaker info
        speaker_txt = os.path.join(self.fg_dir, '..', '..', 'LibriSpeech', 'SPEAKERS.TXT')
        self.speaker_info = self._get_speaker_info(speaker_txt)

    def _get_speaker_info(self, speaker_txt):
        # Read librispeech speaker info to a dataframe
        with open(speaker_txt) as f:
            lines = f.readlines()
        lines = lines[11:]

        # Creae dataframe from lines
        df = pd.DataFrame([l.strip().split('|') for l in lines])

        # Remove extra whitespaces throughout the dataframe
        df = df.iloc[1:]
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        speaker_info = {}
        for i, row in df.iterrows():
            speaker_info[row[df.columns[0]]] = row[df.columns[1]]

        return speaker_info

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
            event_audio_list[1:], event_audio_list[0], multi_ch_seed)

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

        # Source file list and speaker info
        source_files = []
        for obv in jams.annotations[0].data:
            source_files.append(obv.value['source_file'])

        # Pad source files to length 3 for collate_fn
        if len(source_files) == 3:
            source_files.append('None')

        other_spk_info = []
        for sf in source_files[1:]: # skip background
            if sf == 'None':
                other_spk_info.append(('None', 'None'))
                continue
            spk_id = os.path.basename(sf).split('-')[0]
            if spk_id != tgt_id:
                other_spk_info.append((spk_id, self.speaker_info[spk_id]))
        speaker_info = [(tgt_id, self.speaker_info[tgt_id])] + other_spk_info

        inputs = {
            'mixture': mixture,
            'embeds': embeds,
            'enrollments': enrollments,
            'source_files': source_files,
            'speaker_info': speaker_info,
            'embed_paths': embed_paths
        }

        targets = {
            'target': target
        }

        return inputs, targets

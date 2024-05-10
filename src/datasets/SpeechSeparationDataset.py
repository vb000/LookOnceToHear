"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
import logging
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as AT
import scaper
import numpy as np

from src.datasets.MixLibriSpeech import MixLibriSpeechCIPICDataset
from src.datasets.motion_simulator import CIPICMotionSimulator2, RRBRIRMotionSimulator
from src.datasets.multi_ch_simulator import (
    CIPICSimulator, APLSimulator, RRBRIRSimulator, ASHSimulator, CATTRIRSimulator, MultiChSimulator, PRASimulator)

# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")

class SpeechSeparationDataset(Dataset):
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, dset,
                 sr=None, resample_rate=None, num_enroll=10, enroll_len=5, hrtf_type="CIPIC",
                 max_rt60_idx=None, use_bg = False) -> None:
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
        self.use_bg = use_bg

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

        # Get speaker info
        speaker_txt = os.path.join(self.fg_dir, '..', '..', 'LibriSpeech', 'SPEAKERS.TXT')
        self.speaker_info = self._get_speaker_info(speaker_txt)

        # HRTF simulator with motion
        if hrtf_type == 'CIPIC':
            self.multi_ch_simulator = CIPICSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'APL':
            self.multi_ch_simulator = APLSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'ASH':
            self.multi_ch_simulator = ASHSimulator(self.hrtf_list, sr, dset=dset)
        elif hrtf_type == 'CATTRIR':
            self.multi_ch_simulator = CATTRIRSimulator(self.hrtf_list, sr, dset=dset)
        elif hrtf_type == 'RRBRIR':
            self.multi_ch_simulator = RRBRIRSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'MultiCh':
            self.multi_ch_simulator = MultiChSimulator(self.hrtf_list, sr, dset=dset)
        elif hrtf_type == 'CIPIC_MOTION':
            self.multi_ch_simulator = CIPICMotionSimulator2(self.hrtf_list, sr)
        elif hrtf_type == 'PRA':
            self.multi_ch_simulator = PRASimulator(self.hrtf_list, sr)
        else:
            raise NotImplementedError

        # Create speaker to samples mapping
        self.speaker_map = {}
        for i, sample_dir in enumerate(self.samples):
            anns = pd.read_csv(os.path.join(sample_dir, 'mixture.txt'), sep='\t', header=None)
            speaker_ids = anns[2]
            for spk_id in speaker_ids:
                if spk_id not in self.speaker_map:
                    self.speaker_map[spk_id] = [i]
                else:
                    self.speaker_map[spk_id].append(i)

        assert num_enroll == 1, "Only 1 enrollment is supported"

        # Speaker ids
        self.speaker_ids = os.listdir(self.fg_dir)
        self.speaker_ids = [int(x) for x in self.speaker_ids]

    def __len__(self):
        return len(self.samples)

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

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load Audio
        jamsfile = os.path.join(sample_dir, 'mixture.jams')
        _, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        
        chosen_ids = [0, 1]
        if self.dset == 'train':
            chosen_ids = random.sample(list(range(len(event_audio_list[1:]))), k=2)
            
        # Use only 2 speakers
        event_audio_list = [event_audio_list[0], event_audio_list[chosen_ids[0] + 1], event_audio_list[chosen_ids[1] + 1]]
        ann_list = [ann_list[chosen_ids[0]], ann_list[chosen_ids[0]]]

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

        if self.use_bg == False:
            multi_ch_noise *= 0
        
        # Normalize and compute mixture        
        norm_factor = torch.abs(sum(multi_ch_events) + multi_ch_noise).max()
        for i in range(len(multi_ch_events)):
            multi_ch_events[i] /= norm_factor
        multi_ch_noise /= norm_factor
        mixture = sum(multi_ch_events) + multi_ch_noise

        # Target events
        target1 = multi_ch_events[0]
        target2 = multi_ch_events[1]

        # Resample
        mixture = self.resampler(mixture)
        target1 = self.resampler(target1)
        target2 = self.resampler(target2)

        # Choose target index for oracle source selection
        if self.dset == 'train':
            tgt_idx = np.random.randint(2)
        else:
            tgt_idx = 0

        speaker_ids = [spk[-1] for spk in ann_list]

        inputs = {
            'mixture': mixture,
            'speaker_ids': speaker_ids
        }

        targets = {
            'target1': target1,
            'target2': target2,
            'tgt_idx': tgt_idx
        }

        return inputs, targets

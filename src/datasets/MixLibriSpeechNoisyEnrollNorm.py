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
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr

from src.datasets.augmentations import generate_white_noise, generate_pink_noise, generate_brown_noise
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

class MixLibriSpeechNoisyEnroll(Dataset):
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, dset,
                 sr=None, resample_rate=None, num_enroll=10, enroll_len=5, hrtf_type="CIPIC",
                 noise_scale=1.0, max_shift=16, use_motion=False, motion_use_piecewise_arcs=False,
                 augment=False, max_white_noise_level=1e-2, max_pink_noise_level=5e-2,
                 max_brown_noise_level = 5e-2) -> None:
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
        self.noise_scale = noise_scale
        self.max_shift = max_shift
        self.augment = augment
        self.max_white_noise_level = max_white_noise_level
        self.max_pink_noise_level = max_pink_noise_level
        self.max_brown_noise_level = max_brown_noise_level

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
            if use_motion:
                cipic_simulator_type = \
                    lambda sofa, sr: CIPICMotionSimulator2(
                        sofa, sr, use_piecewise_arcs=motion_use_piecewise_arcs)
            else:
                cipic_simulator_type = CIPICSimulator 
            self.multi_ch_simulator = MultiChSimulator(self.hrtf_list, sr, cipic_simulator_type, dset=dset)
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

    def _get_dvector_embedding(self, filename):
        spk_id = filename.split('-')[0]
        embed_map = torch.load(
            os.path.join(self.embed_dir, f'{spk_id}.pt'), map_location='cpu')
        return torch.from_numpy(embed_map[filename]).view(-1)

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
        multi_ch_sim_res = self.multi_ch_simulator.simulate(
            event_audio_list[1:], event_audio_list[0], multi_ch_seed)
        if len(multi_ch_sim_res) == 2:
            multi_ch_events, multi_ch_noise = multi_ch_sim_res
            ang_vels = [0, 0, 0]
        else:
            multi_ch_events, multi_ch_noise, ang_vels = multi_ch_sim_res

        # To torch
        multi_ch_events = [torch.from_numpy(x).float() for x in multi_ch_events]
        multi_ch_noise = torch.from_numpy(multi_ch_noise).float()

        # Scale noise
        if self.dset == 'train':
            noise_scale = random.uniform(*self.noise_scale)
        else:
            _rng = random.Random(idx)
            noise_scale = _rng.uniform(*self.noise_scale)
        multi_ch_noise = multi_ch_noise * noise_scale
        if self.augment and self.dset == 'train' and random.random() < 0.7:
            white_noise = \
                generate_white_noise(multi_ch_noise.shape, self.max_white_noise_level).float()
            pink_noise = \
                generate_pink_noise(multi_ch_noise.shape, self.max_pink_noise_level).float()
            brown_noise = \
                generate_brown_noise(multi_ch_noise.shape, self.max_brown_noise_level).float()

            multi_ch_noise = multi_ch_noise + (white_noise + pink_noise + brown_noise)

        # Normalize and compute mixture
        norm_factor = torch.abs(sum(multi_ch_events) + multi_ch_noise).max()
        if norm_factor > 1.0:
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

        # Angular velocity of target event
        target_ang_vel = ang_vels[tgt_idx] * (180 / np.pi)

        # Input SI-SNR
        input_sisnr = si_snr(target.unsqueeze(0), mixture.unsqueeze(0)).mean()

        # Cross correlation shift for target
        tgt_shift = torch.conv1d(
            target[0:1].unsqueeze(0), target[1:2, self.max_shift:-self.max_shift].unsqueeze(0))
        tgt_shift = tgt_shift[0, 0]
        tgt_shift = torch.argmax(tgt_shift, dim=-1) - self.max_shift

        # Get speaker index for target
        tgt_spk_idx = torch.tensor(self.speaker_ids.index(int(tgt_id)), dtype=torch.long)

        # Source file list and speaker info
        target_source_file = jams.annotations[0].data[tgt_idx + 1].value['source_file']
        source_files = []
        for obv in jams.annotations[0].data:
            source_files.append(obv.value['source_file'])

        # Pad source files to length 3 for collate_fn
        if len(source_files) == 3:
            source_files.append('None')

        # Source embeddings
        source_embeddings = []
        for sf in source_files[1:]:
            if sf == 'None':
                source_embeddings.append(torch.zeros_like(source_embeddings[-1]))
                continue
            filename = os.path.basename(sf)
            source_embeddings.append(self._get_dvector_embedding(filename))

        other_spk_info = []
        for sf in source_files[1:]: # skip background
            if sf == 'None':
                other_spk_info.append(('None', 'None'))
                continue
            spk_id = os.path.basename(sf).split('-')[0]
            if spk_id != tgt_id:
                other_spk_info.append((spk_id, self.speaker_info[spk_id]))
        speaker_info = [(tgt_id, self.speaker_info[tgt_id])] + other_spk_info

        # Select and enrollment sample using the speaker map
        if self.dset == 'train':
            enroll_id = random.choice(self.speaker_map[int(tgt_id)])
        else:
            _rng = random.Random(idx)
            enroll_id = _rng.choice(self.speaker_map[int(tgt_id)])
        enroll_dir = self.samples[enroll_id]

        # Load enrollment audio
        enroll_jams = os.path.join(enroll_dir, 'mixture.jams')
        enroll_spks = pd.read_csv(
            os.path.join(enroll_dir, 'mixture.txt'), sep='\t', header=None)[2].tolist()
        _, enroll_jams, enroll_anns, enroll_event_audio_list = scaper.generate_from_jams(
            enroll_jams, fg_path=self.fg_dir, bg_path=self.bg_dir)
        enroll_source_files = []
        for obv in enroll_jams.annotations[0].data:
            enroll_source_files.append(obv.value['source_file'])
        if len(enroll_source_files) == 3:
            enroll_source_files.append('None')

        # Squeeze to mono
        enroll_event_audio_list = [
            x.squeeze(1) for x in enroll_event_audio_list]

        # Clean enrollment
        enroll_target_idx = enroll_spks.index(int(tgt_id))
        enroll_clean_path = enroll_jams.annotations[0].data[enroll_target_idx + 1].value['source_file']
        enroll_clean_anechoic = torch.from_numpy(enroll_event_audio_list[enroll_target_idx + 1]).float()

        # Ground truth embeddings and negative embeddings
        embedding_gt = self._get_dvector_embedding(os.path.basename(enroll_clean_path))
        embedding_neg = []
        for i, sf in enumerate(enroll_source_files[1:]):
            if sf == 'None':
                embedding_neg.append(torch.zeros_like(embedding_neg[-1]))
                continue
            filename = os.path.basename(sf)
            spk_id = filename.split('-')[0]
            if spk_id != tgt_id:
                embedding_neg.append(self._get_dvector_embedding(filename))

        # Simulate multi-channel audio
        enroll_multi_ch_sim_res = self.multi_ch_simulator.simulate(
            enroll_event_audio_list[1:], enroll_event_audio_list[0], multi_ch_seed,
            face_to_face_idx=enroll_target_idx)
        if len(enroll_multi_ch_sim_res) == 2:
            enroll_multi_ch_events, enroll_multi_ch_noise = enroll_multi_ch_sim_res
            enroll_erros = [0, 0, 0]
        else:
            enroll_multi_ch_events, enroll_multi_ch_noise, enroll_erros = enroll_multi_ch_sim_res
        target_enroll_error = enroll_erros[enroll_target_idx] * (180 / np.pi)
        enroll_multi_ch_events, enroll_multi_ch_noise
        enroll_event_audio_list = [torch.from_numpy(enroll_multi_ch_noise).float()]
        enroll_event_audio_list += [
            torch.from_numpy(x).float() for x in enroll_multi_ch_events
        ]

        # Scale noise
        if self.dset == 'train':
            enroll_noise_scale = random.uniform(*self.noise_scale)
        else:
            _rng = random.Random(idx + 123)
            enroll_noise_scale = _rng.uniform(*self.noise_scale)
        enroll_event_audio_list[0] = enroll_event_audio_list[0] * enroll_noise_scale
        if self.augment and self.dset == 'train' and random.random() < 0.7:
            white_noise = \
                generate_white_noise(multi_ch_noise.shape, self.max_white_noise_level).float()
            pink_noise = \
                generate_pink_noise(multi_ch_noise.shape, self.max_pink_noise_level).float()
            brown_noise = \
                generate_brown_noise(multi_ch_noise.shape, self.max_brown_noise_level).float()

            enroll_event_audio_list[0] += white_noise + pink_noise + brown_noise

        # Normalize and compute enrollment
        enroll_norm_factor = torch.abs(sum(enroll_event_audio_list)).max()
        if enroll_norm_factor > 1.0:
            for i in range(len(enroll_event_audio_list)):
                enroll_event_audio_list[i] /= enroll_norm_factor
        enroll_clean = enroll_event_audio_list[enroll_target_idx + 1]
        enroll = sum(enroll_event_audio_list)

        # Enroll SI-SNR
        enroll_sisnr = si_snr(enroll.unsqueeze(0), enroll_clean.unsqueeze(0)).mean()

        # Resample
        mixture = self.resampler(mixture)
        target = self.resampler(target)
        enroll = self.resampler(enroll)

        inputs = {
            'mixture': mixture,
            'mixture_sisnr': input_sisnr,
            'mixture_embeddings': source_embeddings,
            'enrollments': enroll.unsqueeze(0),
            'enrollments_clean': enroll_clean.unsqueeze(0),
            'enrollments_clean_anechoic': enroll_clean_anechoic.unsqueeze(0),
            'enrollments_clean_path': [enroll_clean_path],
            'enrollments_id': torch.tensor([int(tgt_id)]),
            'enrollments_source_files': enroll_source_files,
            'enrollments_sisnr': enroll_sisnr,
            'tgt_ang_vel': target_ang_vel,
            'tgt_enroll_error': target_enroll_error,
            'tgt_shift': tgt_shift,
            'tgt_idx': torch.tensor(tgt_idx),
            'target_source_file': target_source_file,
            'source_files': source_files,
            'speaker_info': speaker_info
        }

        targets = {
            'target': target,
            'embedding_gt': embedding_gt.unsqueeze(0),
            'embedding_neg': [_.unsqueeze(0) for _ in embedding_neg],
            'tgt_spk_idx': tgt_spk_idx
        }

        return inputs, targets
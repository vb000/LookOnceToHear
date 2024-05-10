import os, glob
import re
import json

import numpy as np
import pandas as pd
import random
import sofa
from scipy.signal import resample, convolve
import soundfile as sf
import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio
from IPython.core.display import display

def _show_rir(rir, sr):
    """
    Args:
        rir (np.ndarray): RIR [2, n_samples]
        sr (int): Sampling rate
    """
    display(Audio(rir, rate=sr))
    plt.figure(figsize=(10, 2))
    plt.plot(rir[0], label='Left')
    plt.plot(rir[1], label='Right')
    plt.show()

class SOFASimulator():
    def __init__(self, sofa_text_file, fs) -> None:
        sofa_dir = os.path.dirname(sofa_text_file)
        with open(sofa_text_file, 'r') as f:
            subject_sofa_list = f.read().split('\n')
            self.sofa_files = [os.path.join(sofa_dir, x) for x in subject_sofa_list]
        self.fs = fs
        self.face_to_face_idx = None
        self.debug = False

    def _convolve(self, src, hrtf, idx):
        ir_idx = {"M": idx}
        _sr = hrtf.Data.SamplingRate.get_values(indices=ir_idx).item()
        rir = hrtf.Data.IR.get_values(indices=ir_idx).astype(np.float32)
        if self.debug and idx == self.face_to_face_idx:
            from IPython.display import Audio
            from IPython.core.display import display
            print("Original RIR")
            display(Audio(rir, rate=_sr))
        rir = torchaudio.functional.resample(torch.from_numpy(rir), _sr, self.fs).numpy()
        if self.debug and idx == self.face_to_face_idx:
            print("RIR")
            display(Audio(rir, rate=self.fs))
            print("SRC")
            display(Audio(src, rate=self.fs))
        src_l = convolve(src, rir[0])[:len(src)]
        src_r = convolve(src, rir[1])[:len(src)]
        bi_src = np.stack([src_l, src_r], axis=0)
        if self.debug and idx == self.face_to_face_idx:
            print("BI_SRC")
            display(Audio(bi_src, rate=self.fs))
        return bi_src

    def simulate(self, srcs, noise, seed=None, face_to_face_idx=None):
        """
        Simulate binaural recordings from monaural recordings using a random HRIR
        obtained from the CIPIC database. Subject as well as HRIR index are
        randomly chosen based on the seed. The results can be made reproducible
        by setting the seed.

        Args:
            srcs ([np.ndarray]): Monaural sources
            noise (np.ndarray): Monaural noise
            face_to_face_idx (int, optional):
                source index to be placed at face to face position. Defaults to None.
            fs (int, optional): Sampling rate. Defaults to 16000.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        rng = random.Random(seed)
        sofa_file = rng.choice(self.sofa_files)
        if self.debug:
            print(sofa_file)

        # Reset random state if face_to_face_idx is not None
        if face_to_face_idx is not None:
            rng = random.Random(seed + 123)

        hrtf = sofa.Database.open(sofa_file)
        bi_srcs = []
        for i, src in enumerate(srcs):
            if face_to_face_idx is not None and i == face_to_face_idx:
                bi_srcs.append(self._convolve(
                    src, hrtf, self.face_to_face_idx))
            else:
                bi_srcs.append(self._convolve(
                    src, hrtf, rng.choice(range(hrtf.Dimensions.M))))
        bi_noise = self._convolve(
            noise, hrtf, rng.choice(range(hrtf.Dimensions.M)))
        return bi_srcs, bi_noise

class CIPICSimulator(SOFASimulator):
    def __init__(self, sofa_text_file, fs) -> None:
        super().__init__(sofa_text_file, fs)
        self.face_to_face_idx = 608

class APLSimulator(SOFASimulator):
    def __init__(self, sofa_text_file, fs) -> None:
        super().__init__(sofa_text_file, fs)
        self.face_to_face_idx = 0

class RRBRIRSimulator(SOFASimulator):
    def __init__(self, sofa_text_file, fs) -> None:
        super().__init__(sofa_text_file, fs)
        self.face_to_face_idx = 18

class ASHSimulator():
    def __init__(self, hrtf_list, fs, dset='train'):
        self.fs = fs

        all_brirs = glob.glob(os.path.join(hrtf_list, "*/*.wav"))

        # Match BRIR_R*_C*_E*_A*.wav among brirs
        brirs = []
        for brir in all_brirs:
            if re.match(r'.*?/BRIR_R.*?_C.*?_E.*?_A.*?.wav', brir):
                brirs.append(brir)

        brir_df = pd.DataFrame(columns=['room', 'config', 'elevation', 'azimuth', 'path'])
        for brir in brirs:
            room, config, elevation, azimuth = re.match(
                r'.*?/BRIR_R(.*?)_C(.*?)_E(.*?)_A(.*?)\.wav', brir).groups()
            brir_df.loc[len(brir_df)] = [room, config, elevation, azimuth, brir]
        brir_df['config'] = brir_df.apply(lambda x: x['room'] + '_' + x['config'], axis=1)

        train_rooms = ['05A', '05B', '06', '07', '09', '12',
                       '13', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                       '26', '27', '28', '31', '32', '33', '34']
        val_rooms = ['03', '04', '08', '10', '11', '30']
        test_rooms = ['01', '02', '14', '15', '16', '29']

        if dset == 'train':
            brir_df = brir_df[brir_df['room'].isin(train_rooms)]
        elif dset == 'val':
            brir_df = brir_df[brir_df['room'].isin(val_rooms)]
        elif dset == 'test':
            brir_df = brir_df[brir_df['room'].isin(test_rooms)]
        else:
            raise ValueError(f"Invalid dset: {dset}")

        # Remove 'room' column
        brir_df = brir_df.drop(columns=['room'])

        # Separate 0 aziuth and others
        brir_df_0 = brir_df[brir_df['azimuth'] == '0'].drop(columns=['elevation', 'azimuth'])
        brir_df_non0 = brir_df[brir_df['azimuth'] != '0'].drop(columns=['elevation', 'azimuth'])

        # Group by config
        self.brir_df_0 = brir_df_0.groupby('config').agg(list).reset_index()
        self.brir_df_non0 = brir_df_non0.groupby('config').agg(list).reset_index()

        # Add count column
        self.brir_df_0['count'] = brir_df_0['path'].apply(lambda x: len(x))
        self.brir_df_non0['count'] = brir_df_non0['path'].apply(lambda x: len(x))

        self.debug = False

    def _convolve(self, src, hrtf_file):
        rir, _sr = torchaudio.load(hrtf_file)
        if self.fs != _sr:
            rir = torchaudio.functional.resample(rir, _sr, self.fs).numpy()
        src_l = convolve(src, rir[0])[:len(src)]
        src_r = convolve(src, rir[1])[:len(src)]
        bi_src = np.stack([src_l, src_r], axis=0)
        return bi_src

    def simulate(self, srcs, noise, seed=None, face_to_face_idx=None):
        """
        Simulate binaural recordings from monaural recordings using a random HRIR
        obtained from the CIPIC database. Subject as well as HRIR index are
        randomly chosen based on the seed. The results can be made reproducible
        by setting the seed.

        Args:
            srcs ([np.ndarray]): Monaural sources
            noise (np.ndarray): Monaural noise
            face_to_face_idx (int, optional):
                source index to be placed at face to face position. Defaults to None.
            fs (int, optional): Sampling rate. Defaults to 16000.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        rng = random.Random(seed)
        room_cfg = rng.choice(self.brir_df_non0['config'])
        if self.debug:
            print(f"Room config: {room_cfg}; face_to_face_idx: {face_to_face_idx}")

        # Reset random state if face_to_face_idx is not None
        if face_to_face_idx is not None:
            rng = random.Random(seed + 123)

        bi_srcs = []
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(projection='polar')
        for i, src in enumerate(srcs):
            if face_to_face_idx is not None and i == face_to_face_idx:
                hrtf_file = rng.choice(
                    self.brir_df_0[self.brir_df_0['config'] == room_cfg]['path'].item())
                bi_srcs.append(self._convolve(src, hrtf_file))
            else:
                hrtf_file = rng.choice(
                    self.brir_df_non0[self.brir_df_non0['config'] == room_cfg]['path'].item())
                bi_srcs.append(self._convolve(src, hrtf_file))
            # hrtf_azimuth = float(re.match(r'.*?/BRIR_R.*?_C.*?_E.*?_A(.*?)\.wav', hrtf_file).groups()[0])
            # _ = ax.scatter([np.pi * hrtf_azimuth / 180], [1.0], label=f"Src {i} (azimuth: {hrtf_azimuth})")
        # plt.legend()
        # plt.title(f"Room config: {room_cfg} (face_to_face_idx: {face_to_face_idx})")
        # plt.show()
        hrtf_file = rng.choice(
            self.brir_df_non0[self.brir_df_non0['config'] == room_cfg]['path'].item())
        bi_noise = self._convolve(noise, hrtf_file)
        return bi_srcs, bi_noise

class CATTRIRSimulator():
    def __init__(self, hrtf_list, fs, dset='train'):
        self.fs = fs
        self.hrtf_list = hrtf_list
        if dset == 'train':
            self.rooms = ['0_0s', '0_1s', '0_2s', '0_5s', '0_6s', '0_7s', '1_0s']
        elif dset == 'val':
            self.rooms = ['0_3s',  '0_9s']
        elif dset == 'test':
            self.rooms = ['0_4s', '0_8s']
        else:
            raise ValueError(f"Invalid dset: {dset}")
        self.azimuths = list(range(-90, 95, 5))
        self.enroll_azimuths_0 = []
        self.enroll_azimuths_non0 = []
        for azimuth in self.azimuths:
            if abs(azimuth) <= 15:
                self.enroll_azimuths_0.append(azimuth)
            else:
                self.enroll_azimuths_non0.append(azimuth)
        self.enroll_azimuths_0 = self.enroll_azimuths_0[1:-1]
        # print(f"Enroll azimuths (0): {self.enroll_azimuths_0}")
        # print(f"Enroll azimuths (non-0): {self.enroll_azimuths_non0}")

    def _convolve(self, src, hrtf_file):
        rir, _sr = torchaudio.load(hrtf_file)
        if self.fs != _sr:
            rir = torchaudio.functional.resample(rir, _sr, self.fs).numpy()
        src_l = convolve(src, rir[0])[:len(src)]
        src_r = convolve(src, rir[1])[:len(src)]
        bi_src = np.stack([src_l, src_r], axis=0)
        return bi_src

    def simulate(self, srcs, noise, seed=None, face_to_face_idx=None):
        """
        Simulate binaural recordings from monaural recordings using a random HRIR
        obtained from the CIPIC database. Subject as well as HRIR index are
        randomly chosen based on the seed. The results can be made reproducible
        by setting the seed.

        Args:
            srcs ([np.ndarray]): Monaural sources
            noise (np.ndarray): Monaural noise
            face_to_face_idx (int, optional):
                source index to be placed at face to face position. Defaults to None.
            fs (int, optional): Sampling rate. Defaults to 16000.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        rng = random.Random(seed)
        room = rng.choice(self.rooms)
        azimuths = self.azimuths

        # Reset random state if face_to_face_idx is not None
        if face_to_face_idx is not None:
            rng = random.Random(seed + 123)
            azimuths = self.enroll_azimuths_non0

        bi_srcs = []
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(projection='polar')
        for i, src in enumerate(srcs):
            if face_to_face_idx is not None and i == face_to_face_idx:
                az = rng.choice(self.enroll_azimuths_0)
            else:
                az = rng.choice(azimuths)
            # _ = ax.scatter([np.pi * az / 180], [1.0], label=f"Src {i} (azimuth: {az})")
            hrtf_file = os.path.join(self.hrtf_list, room, f'CATT_{room}_{az}.wav')
            bi_srcs.append(self._convolve(src, hrtf_file))
        # plt.legend()
        # plt.title(f"Room config: {room} (face_to_face_idx: {face_to_face_idx})")
        # plt.show()
        azs = rng.sample(azimuths, 3)
        bi_noise = []
        for az in azs:
            hrtf_file = os.path.join(self.hrtf_list, room, f'CATT_{room}_{az}.wav')
            _noise = rng.uniform(0.5, 1.0) * self._convolve(noise, hrtf_file)
            bi_noise.append(_noise)

        bi_noise = sum(bi_noise)
        bi_noise = (bi_noise / np.abs(bi_noise).max()) * np.abs(noise).max()

        return bi_srcs, bi_noise

class MultiChSimulator():
    def __init__(self, hrtf_list, fs, cipic_simulator_type=CIPICSimulator, dset='train'):
        cicpic_list, rrbrir_list, ash_list, cattrir_list = hrtf_list
        self.multi_ch_simulators = [
            cipic_simulator_type(cicpic_list, fs),
            RRBRIRSimulator(rrbrir_list, fs),
            ASHSimulator(ash_list, fs, dset=dset),
            CATTRIRSimulator(cattrir_list, fs, dset=dset)
        ]
        self.sampling_counts = [35, 5, 45, 15] # Equivalent to sampling probabilities
        self.fs = fs

    def simulate(self, srcs, noise, seed=None, face_to_face_idx=None):
        rng = random.Random(seed + 246)
        multi_ch_simulator = rng.sample(
            self.multi_ch_simulators, 1, counts=self.sampling_counts)[0]
        return multi_ch_simulator.simulate(srcs, noise, seed, face_to_face_idx)
    
class PRASimulator():
    def __init__(self, hrtf_list, fs, dset='train'):
        self.f2f_max_diff = 15 # 15 degree max angle distance for f2f
        self.fs = fs
        self.hrtf_list = hrtf_list
        self.rooms = sorted(os.listdir(self.hrtf_list))
        
        n = len(self.rooms)
        train_split = int(round(n * 0.7))
        val_split = int(round(n * 0.8))
        
        if dset == 'train':
            self.rooms = self.rooms[:train_split]
        elif dset == 'val':
            self.rooms = self.rooms[train_split:val_split]
        elif dset == 'test':
            self.rooms = self.rooms[val_split:]
        else:
            raise ValueError(f"Invalid dset: {dset}")

    def _convolve(self, src, hrtf_file):
        rir, _sr = torchaudio.load(hrtf_file)
        if self.fs != _sr:
            rir = torchaudio.functional.resample(rir, _sr, self.fs).numpy()
        
        srcs = []
        for i in range(rir.shape[0]):
            srcs.append(convolve(src, rir[i])[:len(src)])
        
        srcs = np.stack(srcs, axis=0)
        return srcs

    def simulate(self, srcs, noise, seed=None, face_to_face_idx=None):
        """
        Simulate binaural recordings from monaural recordings using a random HRIR
        obtained from the CIPIC database. Subject as well as HRIR index are
        randomly chosen based on the seed. The results can be made reproducible
        by setting the seed.

        Args:
            srcs ([np.ndarray]): Monaural sources
            noise (np.ndarray): Monaural noise
            face_to_face_idx (int, optional):
                source index to be placed at face to face position. Defaults to None.
            fs (int, optional): Sampling rate. Defaults to 16000.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        rng = random.Random(seed)
        room = rng.choice(self.rooms)
        
        room_dir = os.path.join(self.hrtf_list, room)
        metadata_path = os.path.join(room_dir, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
        azimuths = np.array(metadata['rir_params']['angles'])
        azimuth_ids = np.arange(azimuths.shape[0])

        # Reset random state if face_to_face_idx is not None
        if face_to_face_idx is not None:
            rng = random.Random(seed + 123)
            azimuth_ids_f2f = np.where( (np.abs(azimuths - 90) % 180) < self.f2f_max_diff )[0].tolist()
            azimuth_ids = np.where( (np.abs(azimuths - 90) % 180) >= self.f2f_max_diff )[0].tolist()
        else:
            azimuth_ids = azimuth_ids.tolist()
        
        mc_srcs = []
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(projection='polar')
        for i, src in enumerate(srcs):
            if face_to_face_idx is not None and i == face_to_face_idx:
                az_idx = rng.choice(azimuth_ids_f2f)
            else:
                az_idx = rng.choice(azimuth_ids)
            # _ = ax.scatter([np.pi * az / 180], [1.0], label=f"Src {i} (azimuth: {az})")
            hrtf_file = os.path.join(self.hrtf_list, room, f'rir_{az_idx:02d}.wav')
            mc_srcs.append(self._convolve(src, hrtf_file))
        # plt.legend()
        # plt.title(f"Room config: {room} (face_to_face_idx: {face_to_face_idx})")
        # plt.show()
        azs = rng.sample(azimuth_ids, 3)
        mc_noise = []
        for az in azs:
            hrtf_file = os.path.join(self.hrtf_list, room, f'rir_{az:02d}.wav')
            _noise = rng.uniform(0.5, 1.0) * self._convolve(noise, hrtf_file)
            mc_noise.append(_noise)

        mc_noise = sum(mc_noise)
        mc_noise = (mc_noise / np.abs(mc_noise).max()) * np.abs(noise).max()

        return mc_srcs, mc_noise
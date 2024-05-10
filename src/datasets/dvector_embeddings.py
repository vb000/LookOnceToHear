"""
Generate NEMO embeddings with titanet with audio samples in the format:

root_dir
|-- subset 0
│   |-- speaker0
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
│   |-- speaker1
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
│   ├── ...
|-- subset 1
│   |-- speaker0
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
|   |-- speaker1
...
"""

import argparse
import os

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm

from resemblyzer import VoiceEncoder, preprocess_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/MixLibriSpeech/librispeech_scaper_fmt')
    parser.add_argument('--output_dir', type=str, default='data/MixLibriSpeech/librispeech_dvector_embeddings')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    # Load model
    speaker_model = VoiceEncoder()
    speaker_model.eval()

    for dset in os.listdir(args.root_dir):
        assert not os.path.exists(os.path.join(args.output_dir, dset))
        os.makedirs(os.path.join(args.output_dir, dset))
        print(f'Processing {dset}...')

        for speaker in tqdm(os.listdir(os.path.join(args.root_dir, dset))):
            embs = {}
            for audio in os.listdir(os.path.join(args.root_dir, dset, speaker)):
                audio_path = os.path.join(args.root_dir, dset, speaker, audio)
                wav = preprocess_wav(audio_path)
                embs[audio] = speaker_model.embed_utterance(wav)

            # Save embeddings in a pickle file
            torch.save(embs, os.path.join(args.output_dir, dset, speaker + '.pt'))

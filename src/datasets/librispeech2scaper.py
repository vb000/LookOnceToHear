"""
Script to convert LibriSpeech dataset to Scaper format. The script
writes symlinks to the audio files in the LibriSpeech dataset to
the output directory.
"""

import os
import argparse

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir', type=str, default='data/MixLibriSpeech',
        help='Path to datsets')

    args = parser.parse_args()

    dsets = ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean']
    for dset in dsets:
        dset_dir = os.path.join('LibriSpeech', dset)
        print("Processing %s..." % dset)
        for speaker in tqdm(os.listdir(os.path.join(args.root_dir, dset_dir))):
            speaker_dir = os.path.join(dset_dir, speaker)
            out_dir = os.path.join(args.root_dir, 'librispeech_scaper_fmt', dset, speaker)
            assert not os.path.exists(out_dir), "Output directory already exists"
            os.makedirs(out_dir)
            for chapter in os.listdir(os.path.join(args.root_dir, speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter)
                for audiofile in os.listdir(os.path.join(args.root_dir, chapter_dir)):
                    if not audiofile.endswith('.flac'):
                        continue
                    audiofile_path = os.path.join(chapter_dir, audiofile)
                    out_path = os.path.join(out_dir, audiofile)
                    rel_path = os.path.join('..', '..', '..', audiofile_path)
                    os.symlink(rel_path, out_path)

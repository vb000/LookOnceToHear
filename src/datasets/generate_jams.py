import os
import random
from types import SimpleNamespace

from tqdm import tqdm
import numpy as np
import scaper

def generate_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create a scaper object
    sc = scaper.Scaper(
        args.duration, args.foreground, args.background, random_state=args.seed)
    sc.protected_labels = []
    sc.ref_db = args.ref_db
    sc.sr = args.sr

    # Generate soundscapes
    print("Generating %d soundscapes with..." % args.num_soundscapes)
    print("Foreground: %s" % args.foreground)
    print("Background: %s" % args.background)
    print("Output: %s" % args.output_dir)
    for n in tqdm(range(args.num_soundscapes)):
        # Reset the event specifications for foreground and background at the
        # beginning of each loop to clear all previously added events
        sc.reset_bg_event_spec()
        sc.reset_fg_event_spec()

        # add background
        sc.add_background(
            label=('choose', args.bg_labels), source_file=('choose', []),
            source_time=('const', 0))

        # Add random number of foreground events
        n_events = np.random.randint(
            args.num_events_min, args.num_events_max + 1)
        for _ in range(n_events):
            sc.add_event(
                label=('choose', args.fg_labels),
                source_file=('choose', []),
                source_time=('const', 0.0),
                event_time=('uniform', 0.0, 1.0),
                event_duration=(
                    'uniform', args.event_duration_min,
                    args.event_duration_max),
                snr=('uniform', args.snr_min, args.snr_max),
                pitch_shift=None,
                time_stretch=None)

        # Generate
        out_path = os.path.join(args.output_dir, '%08d' % n)
        assert not os.path.exists(out_path)
        os.makedirs(out_path)
        audiofile = os.path.join(out_path, 'mixture.wav')
        jamsfile = os.path.join(out_path, 'mixture.jams')
        txtfile = os.path.join(out_path, 'mixture.txt')
        _, _, ann_list, _ = sc.generate(
            audiofile, jamsfile,
            allow_repeated_label=False,
            allow_repeated_source=False,
            reverb=None,
            disable_sox_warnings=True,
            no_audio=not args.write_audio,
            txt_path=txtfile,
            save_isolated_events=True,
            fix_clipping=True,
            disable_instantiation_warnings=True)

datasets = [
    {
        'foreground': 'data/MixLibriSpeech/librispeech_scaper_fmt/test-clean',
        'fg_labels': [],
        'background': 'data/MixLibriSpeech/wham_noise',
        'bg_labels': ['tt'],
        'output_dir': 'data/MixLibriSpeech/jams/test-clean',
        'num_soundscapes': 10000,
        'num_events_min': 2,
        'num_events_max': 3,
        'duration': 5.0,
        'event_duration_min': 5.0,
        'event_duration_max': 5.0,
        'ref_db': -25.0,
        'snr_min': 15.0,
        'snr_max': 25.0,
        'sr': 16000,
        'seed': 1,
        'write_audio': False
    },
    {
        'foreground': 'data/MixLibriSpeech/librispeech_scaper_fmt/dev-clean',
        'fg_labels': [],
        'background': 'data/MixLibriSpeech/wham_noise',
        'bg_labels': ['cv'],
        'output_dir': 'data/MixLibriSpeech/jams/dev-clean',
        'num_soundscapes': 5000,
        'num_events_min': 2,
        'num_events_max': 3,
        'duration': 5.0,
        'event_duration_min': 5.0,
        'event_duration_max': 5.0,
        'ref_db': -25.0,
        'snr_min': 15.0,
        'snr_max': 25.0,
        'sr': 16000,
        'seed': 2,
        'write_audio': False
    },
    {
        'foreground': 'data/MixLibriSpeech/librispeech_scaper_fmt/train-clean-360',
        'fg_labels': [],
        'background': 'data/MixLibriSpeech/wham_noise',
        'bg_labels': ['tr'],
        'output_dir': 'data/MixLibriSpeech/jams/train-clean-360',
        'num_soundscapes': 100000,
        'num_events_min': 2,
        'num_events_max': 3,
        'duration': 5.0,
        'event_duration_min': 5.0,
        'event_duration_max': 5.0,
        'ref_db': -25.0,
        'snr_min': 15.0,
        'snr_max': 25.0,
        'sr': 16000,
        'seed': 42,
        'write_audio': False
    }
]

if __name__ == '__main__': 
    for dataset in datasets:
        dataset_args = SimpleNamespace(**dataset)
        generate_dataset(dataset_args)

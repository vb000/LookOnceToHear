# Look Once to Hear

This repository provides code for the paper, __Look Once to Hear: Target Speech Hearing with Noisy Examples__. __Look once to hear__ is an intelligent hearable system where users choose to hear a target speaker by just looking at them for a few seconds. This paper won an honorable mention 🏆 at CHI 2024.

https://github.com/vb000/LookOnceToHear/assets/16723254/49483e4d-9ebe-4c56-a84e-43c30d1cc3b9

## Setup

    conda create -n ts-hear python=3.9
    conda activate ts-hear
    pip install -r requirements.txt

## Training

Training data includes clean speech, background sounds, head-realted transfer functions (HRTFs) and binaural room impulse responses (BRIRs). We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture is generated on-the-fly, during training or evaluation, using Scaper's `generate_from_jams` function on a `.jams` specification file.

We provide self-contained datasets [here](https://drive.google.com/drive/u/1/folders/1-Jx23GXdjPe33EF5jGZpj6zn-kIm5jHR), with the source `.jams` specifications we used for training. To perform a training run, it is sufficient to download the `.zip` files provided there, unzip the contents to `data/` directory and run this command:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh> [--frac <0.05 (% train/val batches)>]

To resume a partial run:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh>

## Evaluation

Evaluation is done on speech mixture in similar format as training samples. Checkpoints of the embedding model and the target speech hearing (TSH) model are available [here](https://drive.google.com/file/d/1CP0zbZExcqvNLdP9epyhY4fEVp_oQr59/view?usp=sharing).

    python src.ts_hear_test

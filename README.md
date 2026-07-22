# Look Once to Hear

[![Ab](https://img.shields.io/badge/arxiv-abs-green)](https://arxiv.org/abs/2405.06289) [![Gradio demo](https://img.shields.io/badge/arxiv-pdf-green)](https://arxiv.org/pdf/2405.06289)

This repository provides code for the paper, __Look Once to Hear: Target Speech Hearing with Noisy Examples__. __Look Once to Hear__ is an intelligent hearable system where users choose to hear a target speaker by just looking at them for a few seconds. This paper won best paper honorable mention 🏆 at CHI 2024.

https://github.com/vb000/LookOnceToHear/assets/16723254/49483e4d-9ebe-4c56-a84e-43c30d1cc3b9

## Setup

    conda create -n ts-hear python=3.9
    conda activate ts-hear
    pip install -r requirements.txt

## Data preparation

To rebuild all datasets from scratch, follow [data/README.md](data/README.md).

## Training

Training data includes clean speech, background sounds, head-related transfer functions (HRTFs) and binaural room impulse responses (BRIRs). We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture is generated on-the-fly, during training or evaluation, using Scaper's `generate_from_jams` function on a `.jams` specification file.

We provide self-contained datasets [here](https://drive.google.com/file/d/1eLc-CBj4x0S6yL48WEyPocuENo4nDKdp/view?usp=sharing), with the source `.jams` specifications we used for training. To perform a training run, it is sufficient to download `MixLibriSpeech.tar` provided there, extract the contents to `data/` directory and run this command:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh> [--frac <0.05 (% train/val batches)>]

To resume a partial run:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh>

## Evaluation

Evaluation is done on speech mixture in similar format as training samples.

    python -m src.ts_hear_test

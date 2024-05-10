# Look Once to Hear

## Setup

    conda create -n ts-hear python=3.9
    conda activate ts-hear
    pip install -r requirements.txt

## Training

Training data includes clean speech, background sounds, head-realted transfer functions (HRTFs) and binaural room impulse responses (BRIRs). We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture is generated on-the-fly, during training or evaluation, using Scaper's generate_from_jams function on a .jams specification file.

We provide self-contained datasets [here](https://drive.google.com/drive/u/1/folders/1-Jx23GXdjPe33EF5jGZpj6zn-kIm5jHR), with the source `.jams` specifications we used for training. To perform a training run, it is sufficient to download the `.zip` files provided there, unzip the contents to `data/` directory and run this command:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh> [--frac <0.05 (% train/val batches)>]

To resume a partial run:

    python -m src.trainer --config <configs/tsh.json> --run_dir <runs/tsh>

## Evaluation

Evaluation is done on speech mixture in similar format as training samples. Checkpoints of the embedding model and the target speech hearing (TSH) model are available [here](https://drive.google.com/file/d/12dJNZeHWd764cCYnSoCBF9wRtvFuzUuQ/view?usp=sharing).

    PYTHONPATH=. python src/ts_hear_test.py

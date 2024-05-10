# Target Speaker Hearing

## Setup

    conda create -n ts-hear python=3.9
    conda activate ts-hear
    pip install -r requirements.txt

## Training

Download the data [here](https://drive.google.com/drive/u/1/folders/1-Jx23GXdjPe33EF5jGZpj6zn-kIm5jHR) and unzip the contents to `data/` directory.

    # New training run
    python -m src.trainer --config <configs/speakerbeam.json> --run_dir <runs/speakerbeam> [--frac <0.05 (% train/val batches)>]

    # Resume training
    python -m src.trainer --config <runs/speakerbeam/config.json> --run_dir <runs/speakerbeam> --resume

## Evaluation

    python -m src.trainer --config <configs/speakerbeam.json>  --test [--frac <0.05 (% test batches)>]

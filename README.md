# Target Speaker Hearing

### Setup

    conda create -n ts-hear python=3.9
    conda activate ts-hear
    pip install -r requirements.txt

### Setting up motion simulator

    cd motion_simulator && ./build.sh && cd ..    

### Train

    # New training run
    python -m src.trainer --config <configs/speakerbeam.json> --run_dir <runs/speakerbeam> [--frac <0.05 (% train/val batches)>]

    # Resume training
    python -m src.trainer --config <runs/speakerbeam/config.json> --run_dir <runs/speakerbeam> --resume

### Test

    python -m src.trainer --config <configs/speakerbeam.json>  --test [--frac <0.05 (% test batches)>]

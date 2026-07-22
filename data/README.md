# Data preparation

All URLs verified live 2026-07-21. Prereqs: repo env (python 3.9 + `requirements.txt`) plus `sox` and `ffmpeg` binaries on PATH (Scaper needs both).

Target layout (paths hardcoded in `configs/*.json`):
```
data/
├── MixLibriSpeech/
│   ├── LibriSpeech/                        # step 1 (SPEAKERS.TXT is required)
│   ├── librispeech_scaper_fmt/             # step 2
│   ├── wham_noise/{tr,cv,tt}/              # step 3
│   ├── jams/                               # step 4
│   ├── librispeech_dvector_embeddings/     # step 5
│   └── CIPIC/                              # step 6: 45 *.sofa (split lists: in repo)
├── RRBRIR/                                 # step 6: 5 *.sofa (split lists: in repo)
├── ASH-Listening-Set-8.0/BRIRs/            # step 6
└── CATT_RIRs/Binaural/16k/                 # step 6
```

## 1. LibriSpeech

```bash
mkdir -p data/MixLibriSpeech && cd data/MixLibriSpeech      # NOT .../LibriSpeech —
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz   # tarballs already
wget https://www.openslr.org/resources/12/dev-clean.tar.gz         # contain LibriSpeech/
wget https://www.openslr.org/resources/12/test-clean.tar.gz
for f in *.tar.gz; do tar xzf "$f"; done
cd -
```

## 2. Scaper format (symlinks)

```bash
# script's dset list also includes train-clean-100 (unused by configs);
# create a placeholder so the loop doesn't crash, or edit the list in the script
mkdir -p data/MixLibriSpeech/LibriSpeech/train-clean-100
python -m src.datasets.librispeech2scaper --root_dir data/MixLibriSpeech
```

Asserts per-speaker output dirs don't exist — delete
`librispeech_scaper_fmt/<subset>` before re-running a subset.

## 3. WHAM! noise (16 kHz)

```bash
cd data/MixLibriSpeech
# old storage.googleapis.com URL is dead; this is the current mirror from wham.whisper.ai
curl -L -O https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip  # 18.2 GB
unzip wham_noise.zip    # -> wham_noise/{tr:20000, cv:5000, tt:3000} wavs, 37.7 GB
cd -
```

## 4. Generate jams

```bash
python -m src.datasets.generate_jams    # from repo root (relative paths hardcoded)
```

test/dev/train = 10k/5k/100k specs, seeds 1/2/42, WHAM labels tt/cv/tr. Writes
specs only (`write_audio=False`); audio is rendered on the fly during training.

## 5. Speaker embeddings

```bash
python -m src.datasets.dvector_embeddings   # defaults match the layout above
```

Processes every subset under `librispeech_scaper_fmt/`; asserts output dirs
don't exist.

## 6. HRTFs / BRIRs

| Set | Source | Notes |
|---|---|---|
| CIPIC | [sofacoustics.org/data/database/cipic](https://sofacoustics.org/data/database/cipic/) | all 45 `subject_*.sofa` (82 MB); SOFA format, not UC Davis `.mat` |
| RRBRIR | [IoSR-Surrey/RealRoomBRIRs](https://github.com/IoSR-Surrey/RealRoomBRIRs) | the 5 `UniS_*_BRIR_16k.sofa` files (47 MB) |
| ASH | [ShanonPearce/ASH-Listening-Set v8.0](https://github.com/ShanonPearce/ASH-Listening-Set/releases/tag/v8.0) | `ASH-Listening-Set-8.0.zip` (230 MB); unzip at `data/`; covers every room hardcoded in `multi_ch_simulator.py` |
| CATT | [iosr.uk/software](https://iosr.uk/software/index.php) | [`CATT_RIRs.zip`](https://iosr.uk/software/downloads/CATT_RIRs.zip) (71 MB); unzip at `data/` — its `CATT_RIRs/Binaural/16k/<room>/` layout matches the configs as-is (407 wavs, rooms = RT60 `0_0s…1_0s`, azimuths ±90°/5°) |

```bash
cd data/MixLibriSpeech/CIPIC && curl -s https://sofacoustics.org/data/database/cipic/ \
  | grep -oE 'subject_[0-9]+\.sofa' | sort -u \
  | xargs -I{} wget -q https://sofacoustics.org/data/database/cipic/{} && cd -
```

Split lists: the CIPIC/RRBRIR `{train,val,test}_hrtf.txt` files (one SOFA filename per line, resolved relative to the txt's dir — `multi_ch_simulator.py:31-35`) are **committed in this repo**: CIPIC 31/5/9 subjects; RRBRIR train={Room_D,Room_B}, val={Room_C}, test={Room_A}. ASH/CATT need no lists — their room splits are hardcoded in `multi_ch_simulator.py`.


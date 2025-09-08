## Streaming datasets from Hugging Face

This repo includes a simple script to stream datasets from the Hugging Face Hub using the `datasets` library's streaming mode. Useful for building data pipelines (e.g., for distillation) without downloading entire datasets locally.

### Setup

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

### Usage

```bash
# Print a few samples from C4 (English) and stop after 100 total
python scripts/stream_hf_dataset.py c4 --config en --split train --max-samples 100

# Stream from a gated dataset with auth and write to JSONL
$env:HF_TOKEN="hf_..."  # or set HUGGING_FACE_HUB_TOKEN
python scripts/stream_hf_dataset.py allenai/c4 --config en --split train \
  --max-samples 1000 --output out/c4_en_train_1k.jsonl --print-samples 0

# Keep only certain columns
python scripts/stream_hf_dataset.py c4 --config en --split train \
  --max-samples 200 --columns text,url

# Approximate shuffle while streaming
python scripts/stream_hf_dataset.py c4 --config en --split train \
  --max-samples 500 --shuffle-buffer-size 10000 --seed 123
```

### GR00T dataset (file-repo) automatic downloads

The GR00T dataset is a file repository on the Hub, so use the downloader script to fetch specific subset folders.

```bash
# Download one subset folder
python scripts/download_gr00t_subset.py \
  --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --include gr1_arms_waist.CupToDrawer/** \
  --local-dir data/gr00t

# Download multiple subsets at once
python scripts/download_gr00t_subset.py \
  --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --include gr1_arms_waist.CupToDrawer/** gr1_arms_waist.CanToDrawer/** \
  --local-dir data/gr00t

# Exclude patterns (e.g., skip videos)
python scripts/download_gr00t_subset.py \
  --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --include gr1_arms_waist.*/* \
  --exclude "**/*.mp4" \
  --local-dir data/gr00t
```

You can also set `HF_TOKEN` in your environment for gated datasets.

Flags of interest:
- `--hf-token`: supply a token via CLI; otherwise `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars are used if set
- `--columns`: comma-separated list of columns to keep
- `--max-samples`: cap the number of streamed samples (0 means unlimited)
- `--output`: write JSONL to a file
- `--shuffle-buffer-size`: enable approximate shuffling for streaming datasets



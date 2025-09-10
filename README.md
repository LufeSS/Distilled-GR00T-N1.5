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

### On-demand (no full mirror)

List or fetch only matching files directly from the Hub, without mirroring the entire dataset locally.

```bash
# List first 20 JSON files in a specific subset
python scripts/iterate_gr00t_on_demand.py \
  --include gr1_arms_waist.CupToDrawer/** \
  --ext .json --limit 20 --dry-run

# Download matching JSON files to a local dir
python scripts/iterate_gr00t_on_demand.py \
  --include gr1_arms_waist.CupToDrawer/** \
  --ext .json --output-dir data/gr00t_on_demand --workers 8 --retries 3
```

### List repo structure (folders)

```bash
# Top-level folders
python scripts/list_gr00t_structure.py

# Two levels deep, pretty tree
python scripts/list_gr00t_structure.py --depth 2 --tree
```

### Build manifest and deterministic splits

```bash
python scripts/build_manifest_and_splits.py \
  --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --include gr1_arms_waist.*/* \
  --out-dir manifests/gr00t --seed 42 --train 0.95 --val 0.05 --test 0.0
```

### Jetson: Setup and run GR00T N1.5 (PyTorch)

```bash
# Setup (Jetson AGX Orin, JetPack 6.x). Install Jetson torch separately or set JETSON_TORCH_URL
bash scripts/jetson_setup.sh

# Run GR00T N1.5 (requires HF_TOKEN if gated)
export HF_TOKEN="hf_..."
python scripts/run_gr00t.py --model nvidia/GR00T-N1.5-3B --prompt "Describe how to grasp a mug." --max-new-tokens 128
```

If you hit OOM on 32GB devices, enable swap and reduce `--max-new-tokens`. For lower latency and memory, consider TensorRT-LLM conversion (not included here).


Flags of interest:
- `--hf-token`: supply a token via CLI; otherwise `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars are used if set
- `--columns`: comma-separated list of columns to keep
- `--max-samples`: cap the number of streamed samples (0 means unlimited)
- `--output`: write JSONL to a file
- `--shuffle-buffer-size`: enable approximate shuffling for streaming datasets



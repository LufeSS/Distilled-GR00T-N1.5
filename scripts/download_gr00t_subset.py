#!/usr/bin/env python
"""
Automated downloader for file-repo datasets on Hugging Face Hub (e.g., GR00T).

Features:
- Select one or multiple subset folders via --include patterns
- Optional --exclude patterns
- Resumable downloads via huggingface_hub snapshot
- Auth support via --hf-token or env HF_TOKEN/HUGGING_FACE_HUB_TOKEN
- Safe re-runs (only missing files are fetched)

Examples (PowerShell):
  # Download a single subset folder
  python scripts/download_gr00t_subset.py \
    --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --include gr1_arms_waist.CupToDrawer/** \
    --local-dir data/gr00t

  # Download multiple subsets
  python scripts/download_gr00t_subset.py \
    --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --include gr1_arms_waist.CupToDrawer/** gr1_arms_waist.CanToDrawer/** \
    --local-dir data/gr00t

  # Exclude large files like videos or logs
  python scripts/download_gr00t_subset.py \
    --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --include gr1_arms_waist.*/* \
    --exclude "**/*.mp4" "**/*.log" \
    --local-dir data/gr00t
"""

import argparse
import logging
import os
from typing import List, Optional

from huggingface_hub import snapshot_download


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download subset(s) of a file-repo dataset from Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", required=True, help="Dataset repo id, e.g., nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")
    parser.add_argument("--local-dir", default="data/gr00t", help="Local directory to place files")
    parser.add_argument("--include", nargs="*", default=None, help="Allow patterns (e.g., subset folders like gr1_arms_waist.CupToDrawer/**)")
    parser.add_argument("--exclude", nargs="*", default=None, help="Disallow patterns (e.g., **/*.mp4)")
    parser.add_argument("--revision", default=None, help="Specific git revision (branch, tag, or commit)")
    parser.add_argument("--hf-token", default=None, help="HF token. Otherwise HF_TOKEN/HUGGING_FACE_HUB_TOKEN env is used if set")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")

    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    token = resolve_token(args.hf_token)

    os.makedirs(args.local_dir, exist_ok=True)

    logging.info("Downloading from %s to %s", args.dataset, args.local_dir)
    if args.include:
        logging.info("Include patterns: %s", args.include)
    if args.exclude:
        logging.info("Exclude patterns: %s", args.exclude)

    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=args.local_dir,
        revision=args.revision,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        token=token,
        resume_download=True,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()




#!/usr/bin/env python
"""
Stream samples from a Hugging Face dataset using the datasets streaming API.

Examples:
  - Stream first 100 samples from c4 and print a few:
      python scripts/stream_hf_dataset.py c4 --config en --split train --max-samples 100

  - Stream from a gated dataset using an HF token and write to JSONL:
      $env:HF_TOKEN="hf_..."  # PowerShell on Windows
      python scripts/stream_hf_dataset.py allenai/c4 --config en --split train \
        --max-samples 1000 --output out/c4_en_train_1k.jsonl

Notes:
  - Streaming iterates data without downloading the full dataset locally.
  - Shuffling for streaming uses a buffer; it is approximate.
"""

import argparse
import json
import logging
import os
import sys
from typing import Iterable, Dict, Any, Optional, List

import datasets  # pip install datasets


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
        description="Stream a dataset from Hugging Face Hub and optionally write to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core dataset selectors
    parser.add_argument("dataset", type=str, help="Dataset path/name on the Hub, e.g., 'c4' or 'allenai/c4'")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration name (e.g., 'en')")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    parser.add_argument("--data-files", nargs="*", default=None, help="Optional data files (for datasets that accept it)")
    parser.add_argument("--revision", type=str, default=None, help="Git revision of the dataset repo (branch, tag, or commit)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow execution of remote dataset scripts")

    # Auth
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. If not provided, uses HF_TOKEN or HUGGING_FACE_HUB_TOKEN env vars if present.",
    )

    # Streaming behavior
    parser.add_argument("--shuffle-buffer-size", type=int, default=0, help="Shuffle buffer size for approximate streaming shuffle (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for streaming shuffle")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of samples to stream (0 means unlimited)")

    # Output & display
    parser.add_argument("--output", type=str, default=None, help="Path to write JSONL output. If omitted, no file is written.")
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated list of columns to keep (others will be dropped). Defaults to keeping all.",
    )
    parser.add_argument("--print-samples", type=int, default=5, help="Print the first N samples to stdout (0 disables)")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N samples during streaming")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (repeat for more)")

    args = parser.parse_args()
    return args


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return env_token


def open_output(path: Optional[str]):
    if path is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w", encoding="utf-8")


def maybe_select_columns(sample: Dict[str, Any], columns: Optional[List[str]]) -> Dict[str, Any]:
    if not columns:
        return sample
    return {key: sample.get(key) for key in columns}


def stream_dataset(args: argparse.Namespace) -> None:
    configure_logging(args.verbose)

    token = resolve_token(args.hf_token)

    keep_columns: Optional[List[str]] = None
    if args.columns:
        keep_columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    logging.info("Loading dataset '%s' (config=%s, split=%s) in streaming mode...", args.dataset, args.config, args.split)

    try:
        iterable_ds = datasets.load_dataset(
            path=args.dataset,
            name=args.config,
            split=args.split,
            data_files=args.data_files,
            revision=args.revision,
            streaming=True,
            use_auth_token=token,
            trust_remote_code=args.trust_remote_code,
        )
    except TypeError:
        # Some older/newer versions might use a different argument name for token.
        iterable_ds = datasets.load_dataset(
            path=args.dataset,
            name=args.config,
            split=args.split,
            data_files=args.data_files,
            revision=args.revision,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )

    if args.shuffle_buffer_size and args.shuffle_buffer_size > 0:
        logging.info("Applying streaming shuffle with buffer_size=%d, seed=%d", args.shuffle_buffer_size, args.seed)
        iterable_ds = iterable_ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    # Prepare output
    out_fh = open_output(args.output)
    if out_fh:
        logging.info("Writing JSONL to %s", args.output)

    # Iterate
    total = 0
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    try:
        for sample in iterable_ds:
            if keep_columns:
                sample = maybe_select_columns(sample, keep_columns)

            if out_fh:
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if args.print_samples and total < args.print_samples:
                print(json.dumps(sample, ensure_ascii=False, indent=2))

            total += 1
            if args.log_every and total % args.log_every == 0:
                logging.info("Streamed %d samples", total)

            if max_samples is not None and total >= max_samples:
                break
    finally:
        if out_fh:
            out_fh.close()

    logging.info("Done streaming. Total samples: %d", total)


def main() -> None:
    args = parse_args()
    try:
        stream_dataset(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()



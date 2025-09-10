#!/usr/bin/env python
"""
On-demand iterator for GR00T (and other file-repo datasets) via Hugging Face Hub.

Lists and optionally downloads only the files that match your patterns without
mirroring the entire dataset locally.

Examples (Linux/macOS):
  # List JSON files in a subset folder
  python scripts/iterate_gr00t_on_demand.py \
    --include gr1_arms_waist.CupToDrawer/** \
    --ext .json --limit 20 --dry-run

  # Download matching JSON files into a local dir (respects include/exclude)
  python scripts/iterate_gr00t_on_demand.py \
    --include gr1_arms_waist.CupToDrawer/** \
    --ext .json --output-dir data/gr00t_on_demand

Examples (PowerShell on Windows):
  python scripts/iterate_gr00t_on_demand.py `
    --include gr1_arms_waist.CupToDrawer/** `
    --ext .json --limit 20 --dry-run
"""

import argparse
import logging
import os
from typing import Iterable, List, Optional
import fnmatch

from huggingface_hub import HfFileSystem
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


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
        description="On-demand list and fetch files from a Hub dataset file-repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        default="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        help="Dataset repo id on the Hub",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["**/*"],
        help="Glob patterns (relative to repo root) to include, e.g., gr1_arms_waist.CupToDrawer/**",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Glob patterns to exclude (applied after include)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Comma-separated file extensions to keep (e.g., .json,.npz). Case-insensitive.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 means unlimited)")
    parser.add_argument("--dry-run", action="store_true", help="Only list matching files; do not download")
    parser.add_argument("--output-dir", type=str, default=None, help="If set, download files to this directory with repo-relative paths")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file on failure")
    parser.add_argument("--chunk-bytes", type=int, default=1024*1024, help="Read/write chunk size in bytes")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token. Otherwise uses HF_TOKEN/HUGGING_FACE_HUB_TOKEN if set")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")

    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def should_keep(path: str, exclude_patterns: Optional[List[str]], allowed_exts: Optional[List[str]]) -> bool:
    if exclude_patterns:
        for pat in exclude_patterns:
            if fnmatch.fnmatch(path, pat):
                return False
    if allowed_exts:
        lower = path.lower()
        return any(lower.endswith(ext) for ext in allowed_exts)
    return True


def repo_paths(fs: HfFileSystem, dataset: str, include_patterns: List[str]) -> Iterable[str]:
    base = f"datasets/{dataset}"
    for pat in include_patterns:
        # Ensure no leading slash
        pat = pat.lstrip("/")
        full = f"{base}/{pat}"
        for p in fs.glob(full):
            # Keep files only
            if not p.endswith("/"):
                yield p


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    token = resolve_token(args.hf_token)

    fs = HfFileSystem(token=token)

    allowed_exts: Optional[List[str]] = None
    if args.ext:
        allowed_exts = [e.strip().lower() for e in args.ext.split(",") if e.strip()]

    exclude_patterns: Optional[List[str]] = None
    if args.exclude:
        exclude_patterns = [e.strip() for e in args.exclude if e.strip()]

    # Gather and filter file paths
    matched: List[str] = []
    for p in repo_paths(fs, args.dataset, args.include):
        rel = p.split(f"datasets/{args.dataset}/", 1)[-1]
        if should_keep(rel, exclude_patterns, allowed_exts):
            matched.append(p)

    if args.limit and args.limit > 0:
        matched = matched[: args.limit]

    if args.dry_run or not args.output_dir:
        for p in matched:
            print(p)
        logging.info("Total matched: %d", len(matched))
        return

    # Download files in parallel preserving relative paths
    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    def fetch_one(path: str) -> str:
        rel = path.split(f"datasets/{args.dataset}/", 1)[-1]
        dest = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        attempt = 0
        while True:
            try:
                with fs.open(path, "rb") as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(args.chunk_bytes)
                        if not chunk:
                            break
                        dst.write(chunk)
                return rel
            except Exception as e:
                attempt += 1
                if attempt > args.retries:
                    raise
                sleep_s = min(5 * attempt, 15)
                logging.warning("Retry %d for %s after error: %s (sleep %ss)", attempt, rel, e, sleep_s)
                time.sleep(sleep_s)

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(fetch_one, p) for p in matched]
        for fut in as_completed(futures):
            try:
                _rel = fut.result()
            except Exception as e:
                logging.error("Failed to download a file: %s", e)
                continue
            completed += 1
            if completed % 50 == 0:
                logging.info("Downloaded %d/%d files...", completed, len(matched))

    logging.info("Done. Downloaded %d files.", completed)


if __name__ == "__main__":
    main()



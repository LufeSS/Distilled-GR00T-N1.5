#!/usr/bin/env python
"""
Build a deterministic manifest and train/val/test splits for a Hub file-repo dataset.

- Enumerates files via HfFileSystem include/exclude patterns
- Records repo_id, revision, relative path, size
- Deterministic split assignment by hashing rel path with seed
 - Defaults: train=0.95, val=0.05, test=0.0
 - Writes: manifest.jsonl and splits/{train,val,test}.txt

Examples:
  python scripts/build_manifest_and_splits.py \
    --dataset nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --include gr1_arms_waist.*/* \
    --out-dir manifests/gr00t --seed 42 --train 0.9 --val 0.1
"""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

from huggingface_hub import HfFileSystem, get_latest_revision


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
        description="Build manifest and deterministic splits for a Hub dataset file-repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Repo id, e.g., nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")
    parser.add_argument("--include", nargs="*", default=["**/*"], help="Glob include patterns relative to repo root")
    parser.add_argument("--exclude", nargs="*", default=None, help="Glob exclude patterns")
    parser.add_argument("--revision", default=None, help="Specific git revision; defaults to latest on main")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--train", type=float, default=0.95, help="Train ratio (0..1)")
    parser.add_argument("--val", type=float, default=0.05, help="Val ratio (0..1)")
    parser.add_argument("--test", type=float, default=0.0, help="Test ratio (0..1)")
    parser.add_argument("--out-dir", default="manifests", help="Output directory for manifest and splits")
    parser.add_argument("--hf-token", default=None, help="HF token (or use env HF_TOKEN/HUGGING_FACE_HUB_TOKEN)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def repo_files(fs: HfFileSystem, dataset: str, include: List[str]) -> Iterable[str]:
    base = f"datasets/{dataset}"
    for pat in include:
        pat = pat.lstrip("/")
        full = f"{base}/{pat}"
        for p in fs.glob(full):
            if p.endswith("/"):
                continue
            yield p


def is_excluded(rel_path: str, exclude: Optional[List[str]]) -> bool:
    if not exclude:
        return False
    import fnmatch

    for pat in exclude:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def assign_split(rel_path: str, seed: int, train: float, val: float, test: float) -> str:
    # Normalize ratios in case of float error
    total = train + val + test
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0")
    train /= total; val /= total; test /= total

    key = f"{seed}:{rel_path}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    # Take first 8 hex chars as 32-bit int
    r = int(h[:8], 16) / 0xFFFFFFFF
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


@dataclass
class ManifestRow:
    repo_id: str
    revision: str
    rel_path: str
    size: int
    split: str


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    token = resolve_token(args.hf_token)

    fs = HfFileSystem(token=token)

    revision = args.revision or get_latest_revision(repo_id=args.dataset, repo_type="dataset")
    base = f"datasets/{args.dataset}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    split_dir = os.path.join(args.out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    split_fhs = {
        "train": open(os.path.join(split_dir, "train.txt"), "w", encoding="utf-8"),
        "val": open(os.path.join(split_dir, "val.txt"), "w", encoding="utf-8"),
        "test": open(os.path.join(split_dir, "test.txt"), "w", encoding="utf-8"),
    }

    total_files = 0
    total_bytes = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for p in repo_files(fs, args.dataset, args.include):
            rel = p.split(f"{base}/", 1)[-1]
            if is_excluded(rel, args.exclude):
                continue
            try:
                info = fs.info(p)
            except Exception as e:
                logging.warning("Skip (info failed): %s (%s)", p, e)
                continue
            size = int(info.get("size", 0))
            split = assign_split(rel, args.seed, args.train, args.val, args.test)

            row = ManifestRow(
                repo_id=args.dataset,
                revision=revision,
                rel_path=rel,
                size=size,
                split=split,
            )
            mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            split_fhs[split].write(rel + "\n")

            total_files += 1
            total_bytes += size
            if total_files % 1000 == 0:
                logging.info("Listed %d files (%.2f GB)", total_files, total_bytes / 1024 / 1024 / 1024)

    for fh in split_fhs.values():
        fh.close()

    summary = {
        "repo_id": args.dataset,
        "revision": revision,
        "files": total_files,
        "bytes": total_bytes,
        "gb": round(total_bytes / 1024 / 1024 / 1024, 2),
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Done. Files=%d, Size=%.2f GB", total_files, total_bytes / 1024 / 1024 / 1024)


if __name__ == "__main__":
    main()



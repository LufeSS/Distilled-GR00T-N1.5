#!/usr/bin/env python
"""
List the folder structure of a Hugging Face dataset file-repo (e.g., GR00T).

Examples:
  # Top-level folders only
  python scripts/list_gr00t_structure.py

  # Two levels deep, tree view
  python scripts/list_gr00t_structure.py --depth 2 --tree

  # Specify another dataset repo
  python scripts/list_gr00t_structure.py --dataset allenai/c4
"""

import argparse
import logging
import os
from typing import List, Tuple

from huggingface_hub import HfFileSystem


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
        description="List folder structure of a Hub dataset file-repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        help="Dataset repo id on the Hub",
    )
    parser.add_argument("--depth", type=int, default=1, help="How many directory levels to show")
    parser.add_argument("--tree", action="store_true", help="Pretty-print as a tree")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token (or use HF_TOKEN/HUGGING_FACE_HUB_TOKEN env)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def list_dirs(fs: HfFileSystem, base: str, max_depth: int) -> List[Tuple[int, str]]:
    """Breadth-first list of directories up to max_depth relative to base.

    Returns a list of (level, relative_dir_path) where level starts at 1 for
    top-level folders beneath base.
    """
    results: List[Tuple[int, str]] = []
    queue: List[Tuple[int, str]] = [(0, base)]

    while queue:
        level, current = queue.pop(0)
        if level >= max_depth:
            continue

        try:
            entries = fs.ls(current, detail=True)
        except Exception as e:
            logging.warning("Failed to list %s: %s", current, e)
            continue

        for entry in entries:
            name = entry.get("name") or entry  # fsspec returns dicts when detail=True
            entry_type = entry.get("type") if isinstance(entry, dict) else None
            # Normalize directory detection
            is_dir = False
            if isinstance(entry, dict):
                is_dir = entry_type == "directory" or name.endswith("/")
            else:
                is_dir = str(name).endswith("/")

            if not is_dir:
                continue

            # Compute relative path (without trailing slash)
            rel = name[len(base) + 1 :].rstrip("/")
            # Skip the base itself
            if not rel:
                continue
            results.append((level + 1, rel))

            # Queue next level
            queue.append((level + 1, name.rstrip("/")))

    return results


def print_plain(dirs: List[Tuple[int, str]]) -> None:
    for level, rel in dirs:
        print(rel)


def print_tree(dirs: List[Tuple[int, str]]) -> None:
    # Simple indentation-based tree
    for level, rel in dirs:
        indent = "  " * (level - 1)
        # print only the leaf of this level (last component)
        leaf = rel.split("/")[-1]
        print(f"{indent}{leaf}")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    token = resolve_token(args.hf_token)

    fs = HfFileSystem(token=token)
    base = f"datasets/{args.dataset}"

    dirs = list_dirs(fs, base, max_depth=max(1, args.depth))
    if args.tree:
        print_tree(dirs)
    else:
        print_plain(dirs)


if __name__ == "__main__":
    main()



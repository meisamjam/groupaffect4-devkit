"""Entry point: `python -m tools.annotation_gui [BIDS_ROOT]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import run


def main() -> int:
    parser = argparse.ArgumentParser(prog="annotation_gui")
    parser.add_argument(
        "bids_root",
        nargs="?",
        default=None,
        help="Path to a BIDS dataset root (contains sub-*/ses-* folders). "
        "If omitted, open via File → Open BIDS root…",
    )
    args = parser.parse_args()
    root = Path(args.bids_root).resolve() if args.bids_root else None
    if root is not None and not root.is_dir():
        print(f"error: BIDS root not found: {root}", file=sys.stderr)
        return 2
    return run(bids_root=root)


if __name__ == "__main__":
    sys.exit(main())

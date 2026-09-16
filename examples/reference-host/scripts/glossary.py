#!/usr/bin/env python3
"""Write deterministic host terminology for Tradusco."""

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "source/terms.json", args.output)


if __name__ == "__main__":
    main()

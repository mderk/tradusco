#!/usr/bin/env python3
"""Print keys present in every selected built host catalog."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).parents[1]


def main() -> None:
    selected = os.environ.get("TRADUSCO_LANGS", "fr,de").split(",")
    key_sets = [set(json.loads((ROOT / f"dist/{language}.json").read_text())) for language in selected]
    print(json.dumps(sorted(set.intersection(*key_sets))))


if __name__ == "__main__":
    main()

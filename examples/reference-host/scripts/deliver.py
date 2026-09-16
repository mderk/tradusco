#!/usr/bin/env python3
"""Build selected host JSON catalogs from the interchange CSV."""

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).parents[1]


def main() -> None:
    selected = os.environ.get("TRADUSCO_LANGS", "fr,de").split(",")
    with (ROOT / "catalog.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    for language in selected:
        values = {row["en"]: row[language] for row in rows if row.get(language)}
        (output / f"{language}.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract host messages while preserving existing target values."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "catalog.csv"
FIELDS = ["en", "context", "fr", "de"]


def main() -> None:
    existing = {}
    if CATALOG.exists():
        with CATALOG.open(encoding="utf-8", newline="") as file:
            existing = {row["en"]: row for row in csv.DictReader(file)}
    messages = json.loads((ROOT / "source/messages.json").read_text(encoding="utf-8"))
    with CATALOG.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for message in messages:
            previous = existing.get(message["text"], {})
            writer.writerow({
                "en": message["text"],
                # Context is edited in the table and handed back by export: keep it.
                "context": previous.get("context", ""),
                "fr": previous.get("fr", ""),
                "de": previous.get("de", ""),
            })


if __name__ == "__main__":
    main()

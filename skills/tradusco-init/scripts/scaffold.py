#!/usr/bin/env python3
"""Create the minimal Tradusco integration config from an existing host CSV."""

import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True, help="Host repository root")
    parser.add_argument("--csv", type=Path, required=True, help="Existing interchange CSV")
    parser.add_argument("--base", required=True, help="Base-language column")
    parser.add_argument("--locales", required=True, help="Comma-separated target columns")
    args = parser.parse_args()

    host = args.host.resolve()
    source = (host / args.csv).resolve()
    state = host / ".tradusco"
    config_file = state / "config.json"
    if config_file.exists():
        parser.error(f"existing config will not be overwritten: {config_file}")
    if not source.is_file() or not source.is_relative_to(host):
        parser.error("--csv must name an existing file inside the host repository")
    locales = [item.strip() for item in args.locales.split(",")]
    if not locales or any(not item for item in locales) or len(set(locales)) != len(locales) or args.base in locales:
        parser.error("--locales must contain unique, non-empty target columns")
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = [name for name in [args.base, "context", *locales] if name not in columns]
        if missing:
            parser.error(f"CSV is missing columns: {', '.join(missing)}")
        keys = [row[args.base] for row in reader]
    if not keys or any(not key for key in keys) or len(keys) != len(set(keys)):
        parser.error("CSV must contain non-empty, unique base-language text keys")

    tradusco = Path(__file__).resolve().parents[3]
    config = {
        "traduscoRoot": str(tradusco),
        "projectDir": "app",
        "sourceCsv": os.path.relpath(source, state),
        "baseCol": args.base,
        "locales": locales,
        "translate": {"protectLangs": []},
    }
    state.mkdir(parents=True, exist_ok=True)
    with config_file.open("x", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")
    (state / "app").mkdir(exist_ok=True)
    print(config_file)


if __name__ == "__main__":
    main()

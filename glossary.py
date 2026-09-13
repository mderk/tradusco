#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lib.glossary import glossary_coverage, lint_glossary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check translated rows against glossary.json"
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--languages", default="")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.project_dir / "config.json").read_text(encoding="utf-8"))
    glossary_path = args.glossary or args.project_dir / "glossary.json"
    glossary = (
        json.loads(glossary_path.read_text(encoding="utf-8"))
        if glossary_path.exists()
        else {}
    )
    with (args.project_dir / config["sourceFile"]).open(
        encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    if args.coverage:
        coverage = glossary_coverage(
            glossary, (row.get(config["baseLanguage"], "") for row in rows)
        )
        print(json.dumps(coverage, ensure_ascii=False, indent=2))
        return 0
    languages = [value for value in args.languages.split(",") if value] or [
        language
        for language in config["languages"]
        if language != config["baseLanguage"]
    ]
    findings = lint_glossary(glossary, rows, config["baseLanguage"], languages)
    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
    else:
        print(f"glossary findings: {len(findings)}")
        for finding in findings:
            print(
                f"{finding['language']} · {finding['term']}: "
                f"{finding['translation']} (expected {finding['expected']})"
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from po_utils import unescape_po_string


def _normalize_key(text: str, *, collapse_whitespace: bool) -> str:
    s = str(text).replace("\r\n", "\n")
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)
    return s.strip()


def _list_po_files(po_dir: Path) -> list[Path]:
    if not po_dir.exists():
        return []
    return sorted([p for p in po_dir.iterdir() if p.is_file() and p.suffix == ".po"])


def _extract_msgids_from_po(content: str) -> list[str]:
    """
    Extract msgid strings from a PO file (header excluded).

    Note: This intentionally targets the common msgid/msgstr shape used by Lingui
    and many gettext flows. It does not handle msgctxt/msgid_plural as separate keys.
    """
    blocks = [b for b in re.split(r"\n\n+", content) if b.strip()]
    msgids: list[str] = []
    for block in blocks:
        m = re.search(r'msgid\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
        if not m:
            continue
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        msgid = unescape_po_string("".join(parts))
        if not msgid:
            continue  # header
        msgids.append(msgid)
    return msgids


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return (list(reader.fieldnames or []), rows)


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_languages_arg(raw: str) -> list[str]:
    langs = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            langs.append(p)
    return langs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract unique msgid strings from base-locale .po files into translations.csv."
    )
    parser.add_argument("--po-dir", required=True, help="Directory with base-locale .po files")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--base-col", default="en", help="Base language column name (default: en)")
    parser.add_argument(
        "--languages",
        default="en,ru,de,fr,es,it,tr,zh,ja",
        help="Comma-separated CSV columns to create when generating a new file",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate CSV from scratch (ignores existing rows)",
    )
    parser.add_argument(
        "--collapse-whitespace",
        action="store_true",
        default=True,
        help="Collapse whitespace for dedupe comparisons (default: true)",
    )
    parser.add_argument(
        "--no-collapse-whitespace",
        action="store_false",
        dest="collapse_whitespace",
        help="Disable whitespace collapsing (dedupe uses exact whitespace).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    po_dir = Path(args.po_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    base_col: str = args.base_col

    po_files = _list_po_files(po_dir)
    if not po_files:
        raise SystemExit(f"No .po files found under: {po_dir}")

    all_phrases: set[str] = set()
    for po in po_files:
        content = po.read_text(encoding="utf-8")
        for msgid in _extract_msgids_from_po(content):
            all_phrases.add(msgid)

    phrases_sorted = sorted(all_phrases)

    existing_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    existing_keys: set[str] = set()

    if out_csv.exists() and not args.regenerate:
        fieldnames, existing_rows = _read_csv_rows(out_csv)
        if base_col not in fieldnames:
            raise SystemExit(f"Existing CSV missing base column '{base_col}': {out_csv}")
        for row in existing_rows:
            base = row.get(base_col) or ""
            if not base:
                continue
            existing_keys.add(_normalize_key(base, collapse_whitespace=args.collapse_whitespace))

        print(f"Read {len(existing_keys)} existing phrases from CSV file")

    if not fieldnames:
        # Create new file
        fieldnames = _parse_languages_arg(args.languages)
        if base_col not in fieldnames:
            fieldnames = [base_col, *[c for c in fieldnames if c != base_col]]
        if not fieldnames:
            raise SystemExit("--languages produced no columns")

    new_phrases = [
        p
        for p in phrases_sorted
        if _normalize_key(p, collapse_whitespace=args.collapse_whitespace) not in existing_keys
    ]

    if existing_rows and not args.regenerate:
        out_rows = list(existing_rows)
        for phrase in new_phrases:
            row = {c: "" for c in fieldnames}
            row[base_col] = phrase
            out_rows.append(row)
        print(f"Added {len(new_phrases)} new phrases to existing {len(existing_keys)} phrases")
    else:
        out_rows = []
        for phrase in phrases_sorted:
            row = {c: "" for c in fieldnames}
            row[base_col] = phrase
            out_rows.append(row)
        print(f"Created new CSV with {len(out_rows)} phrases")

    if args.dry_run:
        print("dry-run: not writing CSV")
        return 0

    _write_csv_rows(out_csv, fieldnames, out_rows)
    print(
        f"Saved translations.csv with {len(out_rows)} phrases (Mode: {'Regenerate' if args.regenerate else 'Add to existing'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


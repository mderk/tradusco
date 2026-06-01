#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _extract_msgid(block: str) -> str:
    match = re.search(r'msgid\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
    if not match:
        return ""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    # Keep raw escaped form for stable ordering inside file;
    # using unescape isn't strictly required for sorting.
    return "".join(parts)


def sort_po_file(path: Path) -> bool:
    """
    Sort a PO file by msgid (alphabetical).

    Returns True if order changed.
    """
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n{2,}", content)

    header: str | None = None
    entries: list[tuple[str, str]] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        msgid = _extract_msgid(block)
        if msgid == "" and "msgstr" in block:
            header = block
        else:
            entries.append((msgid, block))

    sorted_entries = sorted(entries, key=lambda e: e[0])

    already_sorted = all(entries[i][0] == sorted_entries[i][0] for i in range(len(entries)))
    if already_sorted:
        return False

    out_parts: list[str] = []
    if header:
        out_parts.append(header)
    out_parts.extend([block for _, block in sorted_entries])
    path.write_text("\n\n".join(out_parts) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sort .po files by msgid.")
    parser.add_argument("paths", nargs="+", help="Paths to .po files or directories")
    args = parser.parse_args()

    po_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            po_files.extend(sorted(path.glob("*.po")))
        else:
            po_files.append(path)

    if not po_files:
        raise SystemExit("No .po files found.")

    any_changed = False
    for po in po_files:
        if not po.exists():
            raise SystemExit(f"File not found: {po}")
        changed = sort_po_file(po)
        print(f"{po}: {'sorted' if changed else 'already sorted'}")
        any_changed = any_changed or changed

    return 1 if any_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())


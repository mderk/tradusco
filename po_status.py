#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from po_utils import is_fuzzy, unescape_po_string

_CURLY_TOKEN_RE = re.compile(r"\{[^}]+\}")
_LINGUI_TAG_RE = re.compile(r"</?\d+/?\s*>")


def _extract_curly_tokens(text: str) -> set[str]:
    return set(_CURLY_TOKEN_RE.findall(text))


def _extract_lingui_tags(text: str) -> set[str]:
    return set(_LINGUI_TAG_RE.findall(text))


def _placeholders_match(msgid: str, msgstr: str) -> bool:
    return _extract_curly_tokens(msgid) == _extract_curly_tokens(msgstr) and _extract_lingui_tags(
        msgid
    ) == _extract_lingui_tags(msgstr)


def _parse_po_blocks(content: str) -> list[str]:
    return [b for b in re.split(r"\n\n+", content) if b.strip()]


def _extract_msgid(block: str) -> str | None:
    m = re.search(r'msgid\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
    if not m:
        return None
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return unescape_po_string("".join(parts))


def _extract_msgstr(block: str) -> str:
    m = re.search(r'msgstr\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
    if not m:
        return ""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return "".join(parts)


def status_file(po_path: Path) -> dict:
    content = po_path.read_text(encoding="utf-8")
    blocks = _parse_po_blocks(content)

    total = 0
    missing = 0
    fuzzy = 0
    invalid_placeholders = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        msgid = _extract_msgid(block)
        if msgid is None or msgid == "":
            continue
        total += 1

        fz = is_fuzzy(block)
        if fz:
            fuzzy += 1

        msgstr_raw = _extract_msgstr(block)
        if (not msgstr_raw.strip()) or fz:
            missing += 1
            continue

        msgstr = unescape_po_string(msgstr_raw)
        if not _placeholders_match(msgid, msgstr):
            invalid_placeholders += 1

    return {
        "file": str(po_path),
        "total": total,
        "missing": missing,
        "fuzzy": fuzzy,
        "invalid_placeholders": invalid_placeholders,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check PO status (missing/fuzzy/invalid placeholders)."
    )
    parser.add_argument("--lang", help="Language code (informational)")
    parser.add_argument(
        "--po-dir",
        required=True,
        help="Directory containing .po files (e.g. locale_src/ru)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit 1 if missing>0 or invalid_placeholders>0",
    )
    args = parser.parse_args()

    po_dir = Path(args.po_dir).resolve()
    po_files = sorted(po_dir.glob("*.po"))
    files = [status_file(p) for p in po_files]

    summary = {
        "lang": args.lang,
        "po_dir": str(po_dir),
        "total": sum(f["total"] for f in files),
        "missing": sum(f["missing"] for f in files),
        "fuzzy": sum(f["fuzzy"] for f in files),
        "invalid_placeholders": sum(f["invalid_placeholders"] for f in files),
        "files": files,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"missing={summary['missing']} fuzzy={summary['fuzzy']} invalid_placeholders={summary['invalid_placeholders']} total={summary['total']}"
        )
        for f in files:
            print(
                f"- {Path(f['file']).name}: missing={f['missing']} fuzzy={f['fuzzy']} invalid_placeholders={f['invalid_placeholders']} total={f['total']}"
            )

    if args.fail and (summary["missing"] > 0 or summary["invalid_placeholders"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


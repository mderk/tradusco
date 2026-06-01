#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from po_utils import (
    format_po_msgstr,
    is_fuzzy,
    strip_fuzzy_flag,
    unescape_po_string,
)

_CURLY_TOKEN_RE = re.compile(r"\{[^}]+\}")
_LINGUI_TAG_RE = re.compile(r"</?\d+/?\s*>")


def _extract_curly_tokens(text: str) -> set[str]:
    return set(_CURLY_TOKEN_RE.findall(text))


def _extract_lingui_tags(text: str) -> set[str]:
    return set(_LINGUI_TAG_RE.findall(text))


def _placeholders_match(source: str, translation: str) -> tuple[bool, str]:
    src_tokens = _extract_curly_tokens(source)
    dst_tokens = _extract_curly_tokens(translation)
    if src_tokens != dst_tokens:
        return (
            False,
            f"curly placeholders mismatch: src={sorted(src_tokens)} dst={sorted(dst_tokens)}",
        )

    src_tags = _extract_lingui_tags(source)
    dst_tags = _extract_lingui_tags(translation)
    if src_tags != dst_tags:
        return (
            False,
            f"lingui tags mismatch: src={sorted(src_tags)} dst={sorted(dst_tags)}",
        )

    return True, ""


def _recover_utf8_from_latin1(s: str) -> str | None:
    try:
        recovered = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if recovered == s or "\ufffd" in recovered:
        return None
    try:
        if recovered.encode("utf-8").decode("latin-1") != s:
            return None
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return recovered


def _build_progress_lookup(progress: dict[str, str]) -> dict[str, str]:
    """
    Build a lookup map that includes:
    - original keys
    - recovered UTF-8 keys for mojibake entries
    - newline-normalized keys
    """
    lookup: dict[str, str] = {}
    for k, v in progress.items():
        if not v or not str(v).strip():
            continue
        lookup[k] = v
        lookup[k.replace("\r\n", "\n")] = v

        recovered = _recover_utf8_from_latin1(k)
        if recovered:
            lookup.setdefault(recovered, v)
            lookup.setdefault(recovered.replace("\r\n", "\n"), v)
    return lookup


def _parse_blocks(content: str) -> list[str]:
    # Preserve separators to keep file formatting stable.
    parts = re.split(r"(\n\n+)", content)
    return parts


def _extract_msgid(block: str) -> str | None:
    m = re.search(r'msgid\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
    if not m:
        return None
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return unescape_po_string("".join(parts))


def _extract_msgstr_raw(block: str) -> str:
    m = re.search(r'msgstr\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)', block)
    if not m:
        return ""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return "".join(parts)


def _replace_msgstr(block: str, translation: str) -> str:
    # Find the msgstr line start, replace it with properly formatted msgstr.
    lines = block.split("\n")
    msgstr_idx = None
    for i, line in enumerate(lines):
        if line.startswith("msgstr "):
            msgstr_idx = i
            break
    if msgstr_idx is None:
        return block
    before = "\n".join(lines[:msgstr_idx])
    return before + "\n" + format_po_msgstr(translation)


def apply_progress_to_po_file(
    *,
    po_path: Path,
    progress_lookup: dict[str, str],
    force: bool,
    validate_placeholders: bool,
) -> dict[str, int]:
    content = po_path.read_text(encoding="utf-8")
    parts = _parse_blocks(content)

    updated = 0
    missing_before = 0
    missing_after = 0

    new_parts: list[str] = []
    for part in parts:
        # Keep separators unchanged.
        if part.startswith("\n\n"):
            new_parts.append(part)
            continue

        block = part
        if not block.strip():
            new_parts.append(block)
            continue

        msgid = _extract_msgid(block)
        if msgid is None:
            new_parts.append(block)
            continue

        # Skip header (msgid "")
        if msgid == "":
            new_parts.append(block)
            continue

        fz = is_fuzzy(block)
        msgstr_raw = _extract_msgstr_raw(block)
        msgstr_is_empty = not msgstr_raw.strip()

        needs_update = force or msgstr_is_empty or fz
        if not needs_update:
            new_parts.append(block)
            continue

        missing_before += 1

        translation = progress_lookup.get(msgid) or progress_lookup.get(
            msgid.replace("\r\n", "\n")
        )
        if not translation:
            missing_after += 1
            new_parts.append(block)
            continue

        if validate_placeholders:
            ok, reason = _placeholders_match(msgid, translation)
            if not ok:
                # Skip bad translations; keep missing.
                missing_after += 1
                new_parts.append(block)
                continue

        new_block = _replace_msgstr(block, translation)
        if fz:
            new_block = strip_fuzzy_flag(new_block)
        updated += 1
        new_parts.append(new_block)

    new_content = "".join(new_parts)
    if new_content != content:
        po_path.write_text(new_content, encoding="utf-8")

    return {
        "updated": updated,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Tradusco progress.json translations into .po files."
    )
    parser.add_argument("--lang", required=True, help="Language code (e.g. ru)")
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Tradusco project dir (contains <lang>/progress.json)",
    )
    parser.add_argument(
        "--po-dir",
        required=True,
        help="Directory containing .po files for the language (e.g. locale_src/ru)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already-translated entries (default: only fill missing/fuzzy).",
    )
    parser.add_argument(
        "--no-validate-placeholders",
        action="store_false",
        dest="validate_placeholders",
        help="Disable placeholder/tag validation.",
    )
    parser.set_defaults(validate_placeholders=True)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    po_dir = Path(args.po_dir).resolve()
    progress_path = project_dir / args.lang / "progress.json"

    if not progress_path.exists():
        raise SystemExit(f"progress.json not found: {progress_path}")
    if not po_dir.exists():
        raise SystemExit(f"po dir not found: {po_dir}")

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(progress, dict):
        raise SystemExit(f"Invalid progress.json format (expected object): {progress_path}")

    lookup = _build_progress_lookup(progress)

    files = sorted(po_dir.glob("*.po"))
    results = []
    totals = {"updated": 0, "missing_before": 0, "missing_after": 0}
    for po in files:
        r = apply_progress_to_po_file(
            po_path=po,
            progress_lookup=lookup,
            force=args.force,
            validate_placeholders=args.validate_placeholders,
        )
        results.append({"file": str(po), **r})
        for k in totals:
            totals[k] += r[k]

    out = {
        "lang": args.lang,
        "project_dir": str(project_dir),
        "po_dir": str(po_dir),
        **totals,
        "files": results,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(
                f"{Path(r['file']).name}: updated={r['updated']} missing_before={r['missing_before']} missing_after={r['missing_after']}"
            )
        print(
            f"DONE: updated={totals['updated']} missing_after={totals['missing_after']} (lang={args.lang})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


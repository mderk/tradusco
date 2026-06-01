#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_CURLY_TOKEN_RE = re.compile(r"\{[^}]+\}")
_LINGUI_TAG_RE = re.compile(r"</?\d+/?\s*>")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = list(reader)
        return (list(reader.fieldnames or []), rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
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


def _load_json_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _recover_utf8_from_latin1(s: str) -> str | None:
    try:
        recovered = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if recovered == s:
        return None
    if "\ufffd" in recovered:
        return None
    try:
        if recovered.encode("utf-8").decode("latin-1") != s:
            return None
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return recovered


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


def _bootstrap_progress_if_needed(
    *, project_dir: Path, bootstrap_from: Path, lang: str
) -> bool:
    dst = project_dir / lang / "progress.json"
    if dst.exists():
        try:
            current = _load_json_dict(dst)
            if current:
                return False
        except Exception:
            pass

    src = bootstrap_from / lang / "progress.json"
    if not src.exists():
        return False

    data = _load_json_dict(src)
    if not data:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def _context_from_existing_project_csv(project_csv: Path, base_col: str) -> dict[str, str]:
    if not project_csv.exists():
        return {}

    fields, rows = _read_csv(project_csv)
    if "context" not in fields:
        return {}

    context_by_base: dict[str, str] = {}
    for row in rows:
        base = row.get(base_col, "")
        if not base:
            continue
        ctx = row.get("context", "")
        if ctx:
            context_by_base[base] = ctx
    return context_by_base


def _metadata_from_existing_project_csv(
    project_csv: Path, base_col: str, cols: list[str]
) -> dict[str, dict[str, str]]:
    if not project_csv.exists():
        return {}

    fields, rows = _read_csv(project_csv)
    available = [c for c in cols if c in fields]
    if not available:
        return {}

    meta_by_base: dict[str, dict[str, str]] = {}
    for row in rows:
        base = row.get(base_col, "")
        if not base:
            continue
        meta: dict[str, str] = {}
        for c in available:
            v = row.get(c, "")
            if v:
                meta[c] = v
        if meta:
            meta_by_base[base] = meta
    return meta_by_base


def _build_progress_lookup(progress: dict[str, str]) -> dict[str, str]:
    lookup = dict(progress)
    for k, v in progress.items():
        recovered = _recover_utf8_from_latin1(k)
        if recovered and recovered not in lookup:
            lookup[recovered] = v
    return lookup


def _sanitize_progress_for_phrases(
    *, progress_path: Path, phrases: set[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]], list[tuple[str, str]]]:
    progress = _load_json_dict(progress_path)
    if not progress:
        return {}, {}, []

    sanitized = dict(progress)
    migrated: list[tuple[str, str]] = []

    for k in list(sanitized.keys()):
        recovered = _recover_utf8_from_latin1(k)
        if not recovered or recovered not in phrases:
            continue

        if recovered not in sanitized or not str(sanitized.get(recovered) or "").strip():
            sanitized[recovered] = sanitized[k]
            migrated.append((k, recovered))

        del sanitized[k]

    quarantined: dict[str, dict[str, str]] = {}
    for phrase in list(phrases):
        tr = sanitized.get(phrase)
        if not tr or not str(tr).strip():
            continue
        ok, reason = _placeholders_match(phrase, tr)
        if ok:
            continue
        quarantined[phrase] = {"translation": tr, "reason": reason}
        del sanitized[phrase]

    if sanitized != progress:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if quarantined:
            quarantine_path = progress_path.with_name("progress._quarantine.json")
            quarantine_path.write_text(
                json.dumps(quarantined, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return sanitized, quarantined, migrated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync an extracted phrase CSV into a Tradusco project dir."
    )
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--base-col", default="en", help="Base language column name.")
    parser.add_argument(
        "--context-col",
        default="context",
        help='Optional context column name in source CSV (default: "context").',
    )
    parser.add_argument(
        "--ignore-columns",
        default="context",
        help='Comma-separated non-language columns to ignore when inferring languages (default: "context").',
    )
    parser.add_argument("--bootstrap-from", help="Optional project dir to seed progress.json from.")
    parser.add_argument(
        "--sanitize-progress",
        action="store_true",
        default=True,
        help="Sanitize progress.json for current phrases (migrate mojibake keys, quarantine placeholder-breaking translations). Default: true",
    )
    parser.add_argument(
        "--no-sanitize-progress",
        action="store_false",
        dest="sanitize_progress",
        help="Disable progress sanitization.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    source_csv = Path(args.source_csv).resolve()
    base_col: str = args.base_col
    context_col: str = (args.context_col or "").strip()
    ignore_columns = [c.strip() for c in str(args.ignore_columns or "").split(",") if c.strip()]

    if not source_csv.exists():
        raise SystemExit(f"Source CSV not found: {source_csv}")

    source_fields, source_rows = _read_csv(source_csv)
    if base_col not in source_fields:
        raise SystemExit(f"Invalid source CSV (missing '{base_col}' column): {source_csv}")

    phrases = [row.get(base_col, "") for row in source_rows if row.get(base_col, "") != ""]
    phrases = _dedupe_preserve_order(phrases)
    phrase_set = set(phrases)

    source_row_by_phrase: dict[str, dict[str, str]] = {}
    for row in source_rows:
        phrase = row.get(base_col, "")
        if not phrase:
            continue
        if phrase not in source_row_by_phrase:
            source_row_by_phrase[phrase] = row

    ignore_set = set(ignore_columns)
    if context_col:
        ignore_set.add(context_col)

    # Determine language columns from source (exclude base col + known metadata).
    # Also treat context_<lang> columns as metadata, not languages.
    lang_cols = [
        c
        for c in source_fields
        if c
        and c != base_col
        and c not in ignore_set
        and not str(c).startswith("context_")
    ]

    # Metadata columns to keep in output translations.csv
    meta_cols: list[str] = []
    if context_col:
        meta_cols.append(context_col)
    meta_cols.extend([c for c in source_fields if c and str(c).startswith("context_")])
    meta_cols = _dedupe_preserve_order(meta_cols)

    project_csv = project_dir / "translations.csv"
    existing_meta = _metadata_from_existing_project_csv(project_csv, base_col, meta_cols)

    bootstrap_from = Path(args.bootstrap_from).resolve() if args.bootstrap_from else None
    bootstrap_copied: list[str] = []
    if bootstrap_from and bootstrap_from.exists():
        for lang in lang_cols:
            if _bootstrap_progress_if_needed(
                project_dir=project_dir, bootstrap_from=bootstrap_from, lang=lang
            ):
                bootstrap_copied.append(lang)

    progress_by_lang: dict[str, dict[str, str]] = {}
    progress_lookup_by_lang: dict[str, dict[str, str]] = {}
    quarantined_counts: dict[str, int] = {}
    migrated_counts: dict[str, int] = {}
    for lang in lang_cols:
        progress_path = project_dir / lang / "progress.json"
        progress_by_lang[lang] = _load_json_dict(progress_path)

        if args.sanitize_progress and progress_by_lang[lang]:
            sanitized, quarantined, migrated = _sanitize_progress_for_phrases(
                progress_path=progress_path,
                phrases=phrase_set,
            )
            progress_by_lang[lang] = sanitized
            quarantined_counts[lang] = len(quarantined)
            migrated_counts[lang] = len(migrated)

        progress_lookup_by_lang[lang] = _build_progress_lookup(progress_by_lang[lang])

    out_fields = [base_col, *meta_cols, *lang_cols]
    out_rows: list[dict[str, str]] = []
    for phrase in phrases:
        row: dict[str, str] = {base_col: phrase}
        src_row = source_row_by_phrase.get(phrase, {})

        for c in meta_cols:
            src_v = (src_row.get(c) or "").strip()
            if src_v:
                row[c] = src_row.get(c, "") or ""
            else:
                row[c] = existing_meta.get(phrase, {}).get(c, "") or ""

        for lang in lang_cols:
            row[lang] = progress_lookup_by_lang.get(lang, {}).get(phrase, "") or ""
        out_rows.append(row)

    config = {
        "name": project_dir.name,
        "sourceFile": "translations.csv",
        "languages": [base_col, *lang_cols],
        "baseLanguage": base_col,
        "keyColumn": base_col,
    }

    print(f"source_csv: {source_csv}")
    print(f"project_dir: {project_dir}")
    print(f"phrases: {len(phrases)}")
    if bootstrap_from:
        print(f"bootstrap_from: {bootstrap_from}")
        print(f"bootstrap_copied: {bootstrap_copied or '[]'}")
    if args.sanitize_progress:
        print(f"sanitized_progress_quarantined: {quarantined_counts or '{}'}")
        print(f"sanitized_progress_migrated: {migrated_counts or '{}'}")

    if args.dry_run:
        print("dry-run: not writing config.json / translations.csv")
        return 0

    project_dir.mkdir(parents=True, exist_ok=True)
    _write_json(project_dir / "config.json", config)
    _write_csv(project_csv, out_fields, out_rows)
    print(f"wrote: {project_dir / 'config.json'}")
    print(f"wrote: {project_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


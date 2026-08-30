#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lib.utils import (
    LENGTH_DEFAULTS,
    is_valid_translation as _is_valid_translation,
    length_within_limit as _length_within_limit,
    looks_like_json_artifact as _looks_like_json_artifact,
    placeholders_match as _placeholders_match,
)


def _read_json_dict(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = list(reader)
        return (list(reader.fieldnames or []), rows)


@dataclass
class LangReport:
    lang: str
    status: str
    base_rows: int
    filled_cells: int = 0
    missing_cells: int = 0
    invalid_cells: int = 0
    placeholder_mismatches: int = 0
    length_violations: int = 0

    progress_entries: int | None = None
    progress_invalid_values: int | None = None

    # Per-unique-phrase stats (progress is keyed by base phrase)
    unique_phrases: int = 0
    csv_unique_translated: int = 0
    progress_unique_translated: int | None = None
    csv_not_in_progress: int | None = None
    progress_not_in_csv: int | None = None
    progress_vs_csv_mismatches: int | None = None
    csv_conflicting_translations: int | None = None

    samples: dict[str, list[dict[str, str]]] = field(default_factory=dict)


@dataclass
class ProjectReport:
    project_dir: str
    source_csv: str
    base_col: str
    languages: list[str]
    csv_columns: list[str]

    total_rows: int
    base_rows: int
    unique_phrases: int
    duplicate_base_phrases: int

    key_col: str | None = None
    duplicate_key_values: int = 0

    per_language: list[LangReport] = field(default_factory=list)


def _length_options(config: dict, lang: str) -> dict | None:
    """
    Per-language length settings from config.json:

        "lengthCheck": {
          "enabled": true,
          "maxRatio": 1.9,
          "perLang": { "de": { "maxRatio": 2.2 } }
        }

    Returns ``None`` when the check is switched off for this language.
    """
    section = config.get("lengthCheck")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        return None
    if section.get("enabled") is False:
        return None

    per_lang = section.get("perLang") or {}
    override = per_lang.get(lang) if isinstance(per_lang, dict) else None
    if isinstance(override, dict) and override.get("enabled") is False:
        return None

    opts = {k: v for k, v in section.items() if k in LENGTH_DEFAULTS}
    if isinstance(override, dict):
        opts.update({k: v for k, v in override.items() if k in LENGTH_DEFAULTS})
    return opts


def audit_project(*, project_dir: Path, langs: list[str] | None, max_samples: int) -> ProjectReport:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under: {project_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_col = str(config.get("baseLanguage") or "").strip()
    if not base_col:
        raise SystemExit("Invalid config.json: missing baseLanguage")

    source_csv_name = str(config.get("sourceFile") or "translations.csv")
    source_csv = project_dir / source_csv_name
    if not source_csv.exists():
        raise SystemExit(f"translations CSV not found: {source_csv}")

    config_langs = [str(x) for x in (config.get("languages") or [])]
    if not config_langs:
        raise SystemExit("Invalid config.json: missing languages[]")

    all_langs = [lang for lang in config_langs if lang != base_col]
    if langs:
        allow = set(langs)
        all_langs = [lang for lang in all_langs if lang in allow]

    key_col = str(config.get("keyColumn") or "").strip() or None

    columns, rows = _read_csv(source_csv)
    if base_col not in columns:
        raise SystemExit(
            f"Invalid CSV: missing base column '{base_col}'. Columns: {', '.join(columns)}"
        )

    base_rows = [row for row in rows if (row.get(base_col) or "").strip()]
    phrase_to_idxs: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        phrase = (row.get(base_col) or "").strip()
        if phrase:
            phrase_to_idxs[phrase].append(idx)

    base_phrase_counts = Counter((row.get(base_col) or "").strip() for row in base_rows)
    duplicate_base_phrases = sum(1 for _, c in base_phrase_counts.items() if c > 1)

    duplicate_key_values = 0
    if key_col and key_col in columns:
        key_counts = Counter((row.get(key_col) or "").strip() for row in rows)
        duplicate_key_values = sum(1 for k, c in key_counts.items() if k and c > 1)

    report = ProjectReport(
        project_dir=str(project_dir),
        source_csv=str(source_csv),
        base_col=base_col,
        languages=list(all_langs),
        csv_columns=columns,
        total_rows=len(rows),
        base_rows=len(base_rows),
        unique_phrases=len(phrase_to_idxs),
        duplicate_base_phrases=duplicate_base_phrases,
        key_col=key_col if (key_col in columns if key_col else False) else None,
        duplicate_key_values=duplicate_key_values,
    )

    for lang in all_langs:
        lr = LangReport(lang=lang, status="OK", base_rows=len(base_rows), unique_phrases=len(phrase_to_idxs))
        length_opts = _length_options(config, lang)
        if lang not in columns:
            lr.status = "MISSING_COLUMN"
            report.per_language.append(lr)
            continue

        # CSV stats
        for row in base_rows:
            phrase = (row.get(base_col) or "").strip()
            cell = row.get(lang) or ""
            cell_s = str(cell).strip()
            if not cell_s:
                lr.missing_cells += 1
                if max_samples and len(lr.samples.get("missing_cells", [])) < max_samples:
                    sample = {
                        "key": (row.get(report.key_col) or "").strip()
                        if report.key_col
                        else (row.get("key") or "").strip(),
                        "phrase": phrase,
                    }
                    lr.samples.setdefault("missing_cells", []).append(sample)
                continue

            if _looks_like_json_artifact(cell_s):
                lr.invalid_cells += 1
                if max_samples and len(lr.samples.get("invalid_cells", [])) < max_samples:
                    sample = {
                        "key": (row.get(report.key_col) or "").strip()
                        if report.key_col
                        else (row.get("key") or "").strip(),
                        "phrase": phrase,
                        "value": cell_s,
                    }
                    lr.samples.setdefault("invalid_cells", []).append(sample)
                continue

            lr.filled_cells += 1

            ok, reason = _placeholders_match(phrase, cell_s)
            if not ok:
                lr.placeholder_mismatches += 1
                if max_samples and len(lr.samples.get("placeholder_mismatches", [])) < max_samples:
                    sample = {
                        "key": (row.get(report.key_col) or "").strip()
                        if report.key_col
                        else (row.get("key") or "").strip(),
                        "phrase": phrase,
                        "value": cell_s,
                        "reason": reason,
                    }
                    lr.samples.setdefault("placeholder_mismatches", []).append(sample)

            if length_opts is not None:
                fits, why = _length_within_limit(phrase, cell_s, length_opts)
                if not fits:
                    lr.length_violations += 1
                    if max_samples and len(lr.samples.get("length_violations", [])) < max_samples:
                        sample = {
                            "key": (row.get(report.key_col) or "").strip()
                            if report.key_col
                            else (row.get("key") or "").strip(),
                            "phrase": phrase,
                            "value": cell_s,
                            "reason": why,
                        }
                        lr.samples.setdefault("length_violations", []).append(sample)

        # Progress stats
        progress_path = project_dir / lang / "progress.json"
        progress = _read_json_dict(progress_path)
        if progress is None:
            lr.progress_entries = None
            lr.progress_invalid_values = None
            lr.progress_unique_translated = None
            lr.csv_not_in_progress = None
            lr.progress_not_in_csv = None
            lr.progress_vs_csv_mismatches = None
            lr.csv_conflicting_translations = None
            report.per_language.append(lr)
            continue

        lr.progress_entries = len(progress)
        lr.progress_invalid_values = sum(1 for v in progress.values() if _looks_like_json_artifact(v))

        # Compare progress <-> CSV per unique phrase
        csv_not_in_progress = 0
        progress_not_in_csv = 0
        mismatches = 0
        conflicts = 0

        progress_unique_translated = 0
        for phrase, idxs in phrase_to_idxs.items():
            csv_vals = {str(rows[i].get(lang) or "").strip() for i in idxs}
            csv_vals_valid = {v for v in csv_vals if _is_valid_translation(v)}

            if csv_vals_valid:
                lr.csv_unique_translated += 1

            pval = progress.get(phrase)
            pval_valid = _is_valid_translation(pval)
            if pval_valid:
                progress_unique_translated += 1

            if csv_vals_valid and not pval_valid:
                csv_not_in_progress += 1
            if pval_valid and not csv_vals_valid:
                progress_not_in_csv += 1

            if len(csv_vals_valid) > 1:
                conflicts += 1

            if pval_valid and csv_vals_valid and (pval not in csv_vals_valid):
                mismatches += 1
                if max_samples and len(lr.samples.get("progress_vs_csv_mismatches", [])) < max_samples:
                    sample = {
                        "phrase": phrase,
                        "progress": str(pval),
                        "csv_values": "; ".join(sorted(csv_vals_valid))[:400],
                    }
                    lr.samples.setdefault("progress_vs_csv_mismatches", []).append(sample)

        lr.progress_unique_translated = progress_unique_translated
        lr.csv_not_in_progress = csv_not_in_progress
        lr.progress_not_in_csv = progress_not_in_csv
        lr.progress_vs_csv_mismatches = mismatches
        lr.csv_conflicting_translations = conflicts

        report.per_language.append(lr)

    return report


def _print_report(report: ProjectReport) -> None:
    print(
        f"project_dir: {report.project_dir}\n"
        f"source_csv: {report.source_csv}\n"
        f"base_col: {report.base_col}\n"
        f"rows: total={report.total_rows} base_non_empty={report.base_rows} unique_phrases={report.unique_phrases}\n"
        f"duplicates: base_phrases={report.duplicate_base_phrases} key_values={report.duplicate_key_values}\n"
    )

    for lr in report.per_language:
        if lr.status != "OK":
            print(f"[{lr.lang}] status={lr.status}")
            continue
        print(
            f"[{lr.lang}] missing_cells={lr.missing_cells} invalid_cells={lr.invalid_cells} "
            f"placeholder_mismatches={lr.placeholder_mismatches} "
            f"length_violations={lr.length_violations} "
            f"progress_entries={lr.progress_entries} progress_invalid={lr.progress_invalid_values} "
            f"csv_unique={lr.csv_unique_translated} progress_unique={lr.progress_unique_translated} "
            f"csv_not_in_progress={lr.csv_not_in_progress} progress_not_in_csv={lr.progress_not_in_csv} "
            f"progress_vs_csv_mismatches={lr.progress_vs_csv_mismatches} csv_conflicts={lr.csv_conflicting_translations}"
        )

    problem_langs: list[LangReport] = []
    for lr in report.per_language:
        if lr.status != "OK":
            problem_langs.append(lr)
            continue
        if lr.missing_cells or lr.invalid_cells or lr.placeholder_mismatches:
            problem_langs.append(lr)

    if problem_langs:
        print("\nproblem_langs:")
        for lr in sorted(problem_langs, key=lambda x: (-int(x.missing_cells), -int(x.invalid_cells), x.lang)):
            if lr.status != "OK":
                print(f"- {lr.lang}: status={lr.status}")
            else:
                print(
                    f"- {lr.lang}: missing_cells={lr.missing_cells} invalid_cells={lr.invalid_cells} "
                    f"placeholder_mismatches={lr.placeholder_mismatches}"
                )

    # Reported apart from the hard errors above: an over-long label is a layout
    # risk to look at, not a broken build.
    long_langs = [lr for lr in report.per_language if lr.status == "OK" and lr.length_violations]
    if long_langs:
        print("\nlong_labels:")
        for lr in sorted(long_langs, key=lambda x: -int(x.length_violations)):
            print(f"- {lr.lang}: length_violations={lr.length_violations}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a Tradusco project for completeness and correctness."
    )
    parser.add_argument(
        "--project-dir", required=True, help="Path to the Tradusco project directory"
    )
    parser.add_argument(
        "--langs",
        help="Comma-separated list of locales to audit (default: all from config.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Max sample items per category (default: 10, 0 disables samples)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON (machine-readable)",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit with non-zero code if issues are found",
    )
    parser.add_argument(
        "--fail-on-length",
        action="store_true",
        help="With --fail, also treat over-long UI labels as issues",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    langs = [s.strip() for s in str(args.langs).split(",") if s.strip()] if args.langs else None

    report = audit_project(project_dir=project_dir, langs=langs, max_samples=args.max_samples)

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    if not args.fail:
        return 0

    issues = False
    for lr in report.per_language:
        if lr.status != "OK":
            issues = True
            break
        if lr.missing_cells or lr.invalid_cells or lr.placeholder_mismatches:
            issues = True
            break
        if args.fail_on_length and lr.length_violations:
            issues = True
            break

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())


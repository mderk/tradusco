#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

from lib.utils import placeholders_match, validate_translation_text


def read_json(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        temporary = Path(f.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
        temporary = Path(f.name)
    os.replace(temporary, path)


def revision(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def writer_lock(project: Path):
    lock = project / ".run.lock"
    owner = os.environ.get("TRADUSCO_LOCK_PID")
    if lock.exists():
        if lock.read_text(encoding="utf-8").strip() != owner:
            raise SystemExit(f"project is locked: {lock}")
        yield
        return
    try:
        with lock.open("x", encoding="utf-8") as file:
            file.write(f"{os.getpid()}\n")
    except FileExistsError:
        raise SystemExit(f"project is locked: {lock}") from None
    previous = os.environ.get("TRADUSCO_LOCK_PID")
    os.environ["TRADUSCO_LOCK_PID"] = str(os.getpid())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRADUSCO_LOCK_PID", None)
        else:
            os.environ["TRADUSCO_LOCK_PID"] = previous
        lock.unlink(missing_ok=True)


class State:
    def __init__(self, config_path: Path):
        integration = read_json(config_path, {})
        self.root = config_path.parent
        self.project = (self.root / integration.get("projectDir", "project")).resolve()
        config = read_json(self.project / "config.json", {})
        self.project_csv = self.project / config.get("sourceFile", "translations.csv")
        self.catalog = (self.root / integration.get("sourceCsv", self.project_csv)).resolve()
        self.base = config.get("baseLanguage", integration.get("baseCol", "en"))
        self.languages = [x for x in config.get("languages", []) if x != self.base]
        self.editorial_file = self.project / "editorial.json"


def indexed(path: Path, base: str):
    fields, rows = read_csv(path)
    if base not in fields:
        raise SystemExit(f"missing {base} column: {path}")
    by_source: dict[str, dict[str, str]] = {}
    for row in rows:
        source = row.get(base, "")
        if not source:
            continue
        if source in by_source:
            raise SystemExit(f"duplicate source key: {source!r}")
        by_source[source] = row
    return fields, rows, by_source


def validate(source: str, translation: str) -> None:
    ok, reason = validate_translation_text(translation)
    if ok:
        ok, reason = placeholders_match(source, translation)
    if not ok:
        raise SystemExit(f"invalid translation for {source!r}: {reason}")


def apply_edits(state: State, edits: list[dict], write: bool) -> dict[str, int]:
    with writer_lock(state.project) if write else nullcontext():
        fields, rows, by_source = indexed(state.project_csv, state.base)
        editorial = read_json(state.editorial_file, {})
        progress = {lang: read_json(state.project / lang / "progress.json", {}) for lang in state.languages}
        stats = {"applied": 0, "already": 0, "conflict": 0, "missing": 0}
        accepted: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for edit in edits:
            lang = str(edit.get("language", edit.get("lang", "")))
            source = str(edit.get("source", edit.get("key", "")))
            cell = (source, lang)
            if cell in seen:
                raise SystemExit(f"duplicate edit: {source!r}/{lang}")
            seen.add(cell)
            if "from" not in edit:
                raise SystemExit(f"edit requires from: {source!r}/{lang}")
            if lang not in state.languages or lang not in fields:
                raise SystemExit(f"unknown language: {lang}")
            if source not in by_source:
                stats["missing"] += 1
                continue
            current, target = by_source[source].get(lang, ""), str(edit.get("to", ""))
            validate(source, target)
            if current not in (str(edit["from"]), target):
                stats["conflict"] += 1
                continue
            stats["already" if current == target else "applied"] += 1
            accepted.append((lang, source, target))
        if not write or not accepted:
            return stats
        for lang, source, target in accepted:
            editorial.setdefault(lang, {})[source] = target
            progress[lang][source] = target
            by_source[source][lang] = target
        # Durable decision first; rerunning repairs either following file after interruption.
        write_json(state.editorial_file, editorial)
        for lang in {lang for lang, _, _ in accepted}:
            write_json(state.project / lang / "progress.json", progress[lang])
        write_csv(state.project_csv, fields, rows)
        return stats


def differences(state: State, direction: str):
    project_fields, _, project = indexed(state.project_csv, state.base)
    catalog_fields, _, catalog = indexed(state.catalog, state.base)
    missing = [lang for lang in state.languages if lang not in project_fields or lang not in catalog_fields]
    if missing:
        raise SystemExit(f"missing language columns: {', '.join(missing)}")
    changes = []
    shared = project.keys() & catalog.keys()
    for source in shared:
        for lang in state.languages:
            project_value, catalog_value = project[source].get(lang, ""), catalog[source].get(lang, "")
            if direction == "back-sync" and catalog_value and catalog_value != project_value:
                changes.append({"source": source, "language": lang, "from": project_value, "to": catalog_value})
            if direction == "export" and project_value and project_value != catalog_value:
                changes.append({"source": source, "language": lang, "from": catalog_value, "to": project_value})
    return changes, len(catalog.keys() - project.keys())


def sync(state: State, command: str, write: bool, expected: str | None) -> dict:
    with writer_lock(state.project) if write else nullcontext():
        current_revision = revision(state.project_csv, state.catalog)
        changes, unmanaged = differences(state, command)
        result = {"revision": current_revision, "changes": len(changes), "unmanaged": unmanaged, "edits": changes[:20]}
        if not write:
            return result
        if expected != current_revision:
            raise SystemExit("input revision changed; preview again")
        if command == "back-sync":
            result.update(apply_edits(state, changes, True))
            return result
        fields, rows, catalog = indexed(state.catalog, state.base)
        for edit in changes:
            catalog[edit["source"]][edit["language"]] = edit["to"]
        write_csv(state.catalog, fields, rows)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Review and deliver source-as-key translations.")
    parser.add_argument("command", choices=("read", "apply", "back-sync", "export"))
    parser.add_argument("--config", default=".tradusco/config.json")
    parser.add_argument("--source")
    parser.add_argument("--edits")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--expect")
    args = parser.parse_args()
    state = State(Path(args.config).resolve())
    if args.command == "read":
        if not args.source:
            raise SystemExit("read requires --source")
        _, _, rows = indexed(state.project_csv, state.base)
        if args.source not in rows:
            raise SystemExit(f"source not found: {args.source!r}")
        print(json.dumps(rows[args.source], ensure_ascii=False, indent=2))
    elif args.command == "apply":
        if not args.edits:
            raise SystemExit("apply requires --edits")
        value = read_json(Path(args.edits), None)
        if not isinstance(value, list):
            raise SystemExit("edits must be a JSON array")
        result = apply_edits(state, value, args.write)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["conflict"] or result["missing"]:
            return 1
    else:
        print(json.dumps(sync(state, args.command, args.write, args.expect), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

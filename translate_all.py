#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from lib.utils import is_valid_translation


@dataclass
class Job:
    lang: str
    cmd: list[str]


def _load_config(project_dir: Path) -> dict:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under: {project_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _parse_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _looks_invalid(v: object) -> bool:
    # Empty or JSON scaffolding -> a cell that still needs (re)translation.
    return not is_valid_translation(v)


def _langs_with_missing_cells(project_dir: Path, *, langs: list[str], base_col: str) -> list[str]:
    cfg = _load_config(project_dir)
    csv_name = str(cfg.get("sourceFile") or "translations.csv")
    csv_path = project_dir / csv_name
    cols, rows = _parse_csv(csv_path)
    base_rows = [row for row in rows if (row.get(base_col) or "").strip()]

    out: list[str] = []
    for lang in langs:
        if lang not in cols:
            out.append(lang)
            continue
        missing = 0
        for row in base_rows:
            v = row.get(lang)
            if _looks_invalid(v):
                missing += 1
        if missing:
            out.append(lang)
    return out


async def _run_job(job: Job) -> int:
    proc = await asyncio.create_subprocess_exec(
        *job.cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
    )

    assert proc.stdout is not None
    prefix = f"[{job.lang}] "
    async for raw in proc.stdout:
        try:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
        except Exception:
            line = repr(raw)
        print(prefix + line, flush=True)
    return await proc.wait()


async def _run_jobs_with_limit(jobs: list[Job], *, parallel: int) -> dict[str, int]:
    sem = asyncio.Semaphore(max(1, parallel))
    results: dict[str, int] = {}

    async def runner(job: Job) -> None:
        async with sem:
            code = await _run_job(job)
            results[job.lang] = code

    await asyncio.gather(*(runner(j) for j in jobs))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate all locales in a Tradusco project (optionally with fallback)."
    )
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-max-tokens", type=int, default=2048)
    parser.add_argument("--method", default="auto")
    parser.add_argument("--langs", help="Comma-separated subset of locales to run")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Before each pass, detect which locales still have missing/invalid cells and run only those.",
    )
    args = parser.parse_args()

    # Ensure we get live logs even when stdout is redirected.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py3.7+
    except Exception:
        pass

    project_dir = Path(args.project_dir).resolve()
    cfg = _load_config(project_dir)
    base_col = str(cfg.get("baseLanguage") or "en")
    cfg_langs = [str(x) for x in (cfg.get("languages") or [])]
    langs = [lang for lang in cfg_langs if lang != base_col]
    if args.langs:
        allow = {s.strip() for s in str(args.langs).split(",") if s.strip()}
        langs = [lang for lang in langs if lang in allow]
    if not langs:
        raise SystemExit("No destination languages selected.")

    python = sys.executable

    def build_jobs(model: str, run_langs: list[str]) -> list[Job]:
        jobs: list[Job] = []
        for lang in run_langs:
            cmd = [
                python,
                "-u",
                str((Path(__file__).resolve().parent / "translate.py").resolve()),
                "-p",
                str(project_dir),
                "-l",
                lang,
                "-m",
                model,
                "--method",
                str(args.method),
                "-b",
                str(args.batch_size),
                "--batch-max-tokens",
                str(args.batch_max_tokens),
            ]
            jobs.append(Job(lang=lang, cmd=cmd))
        return jobs

    async def run_pass(model: str, label: str) -> dict[str, int]:
        run_langs = langs
        if args.only_missing:
            run_langs = _langs_with_missing_cells(project_dir, langs=langs, base_col=base_col)
            print(f"{label}: locales_to_run={run_langs}", flush=True)
        jobs = build_jobs(model, run_langs)
        if not jobs:
            print(f"{label}: nothing to do", flush=True)
            return {}
        return await _run_jobs_with_limit(jobs, parallel=args.parallel)

    # Pass 1
    results_1 = asyncio.run(run_pass(args.model, "primary"))
    failed_1 = {k: v for k, v in results_1.items() if v != 0}

    # Pass 2 (fallback)
    results_2: dict[str, int] = {}
    failed_2: dict[str, int] = {}
    if args.fallback_model:
        results_2 = asyncio.run(run_pass(args.fallback_model, "fallback"))
        failed_2 = {k: v for k, v in results_2.items() if v != 0}

    if failed_1 or failed_2:
        print("\nFailures:", flush=True)
        for lang, code in sorted({**failed_1, **failed_2}.items()):
            print(f"- {lang}: exit_code={code}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

import tiktoken

from lib.envelope import EnvelopeBuilder
from lib.PromptManager import PromptManager
from lib.storage.filesystem import FileSystemStorageAdapter
from lib.TranslationTool import language_ref, TranslationTool


REVIEWED_LANGUAGES = (
    "fr",
    "ru",
    "it",
    "de",
    "es",
    "es-la",
    "ja",
    "ko",
    "pl",
    "pt-br",
    "pt-pt",
    "zh-cn",
)
PROMPT_LANGUAGES = (
    "fr",
    "it",
    "de",
    "es",
    "es-la",
    "ko",
    "pl",
    "pt-br",
    "pt-pt",
    "zh-cn",
    "zh-tw",
    "tr",
    "uk",
    "th",
    "cs",
    "hu",
    "vi",
    "ro",
    "ar",
    "nl",
)


def round_up(value: float) -> float:
    return math.ceil(value * 100) / 100


def count_batches(
    token_counts: list[int], batch_size: int, languages: int, ratio: float, budget: int
) -> int:
    batches = batch_phrases = batch_tokens = 0
    for tokens in token_counts:
        if batch_phrases and (batch_tokens + tokens) * languages * ratio > budget:
            batches += 1
            batch_phrases = batch_tokens = 0
        batch_phrases += 1
        batch_tokens += tokens
        if batch_phrases >= batch_size or batch_tokens * languages * ratio >= budget:
            batches += 1
            batch_phrases = batch_tokens = 0
    return batches + bool(batch_phrases)


async def measure_prompt(
    *,
    project_path: Path,
    glossary_path: Path,
    rows: list[dict[str, str]],
    row_count: int,
    dst_languages: list[str],
    reference_languages: list[str],
    encoding,
) -> None:
    storage = FileSystemStorageAdapter(project_path)
    config = await storage.load_config(project_path.name)
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    builder = EnvelopeBuilder(
        glossary,
        rows,
        config.baseLanguage,
        dst_languages,
        reference_languages,
    )
    phrases: list[tuple[str, str | None]] = []
    phrase_indices: dict[str, int] = {}
    for index, row in enumerate(rows):
        phrase = row.get(config.baseLanguage, "")
        if not phrase:
            continue
        phrases.append((phrase, row.get("context") or None))
        phrase_indices[phrase] = index
        if len(phrases) == row_count:
            break

    context_parts = await storage.load_context(project_path.name)
    for language in dst_languages:
        language_parts = await storage.load_context(project_path.name, language)
        if language_parts:
            context_parts.append(f"[{language}]\n" + "\n\n".join(language_parts))
    manager = PromptManager(storage, project_path.name)
    tool = TranslationTool(manager)
    batch = builder.build(phrases, phrase_indices)
    prompt = await tool.create_prompt(
        phrases,
        config.baseLanguage,
        [language_ref(language) for language in dst_languages],
        await manager.load_prompt("translation"),
        "\n\n".join(context_parts),
        batch,
    )
    if prompt is None:
        raise RuntimeError("Prompt assembly failed")
    payload = batch.model_dump(exclude_none=True)
    phrase_payloads = payload["phrases"]
    print("\n| Prompt rows | Languages | Chars | Tokens | Glossary | References | Examples |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {len(phrases)} | {len(dst_languages)} | {len(prompt)} | "
        f"{len(encoding.encode(prompt))} | {len(payload.get('glossary', []))} | "
        f"{sum(bool(phrase.get('reference')) for phrase in phrase_payloads)} | "
        f"{sum(len(phrase.get('examples', [])) for phrase in phrase_payloads)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure translation/output token ratios from a reviewed CSV."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--base-language", default="en")
    parser.add_argument("--benchmark-rows", type=int, default=0)
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--prompt-project", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--prompt-rows", type=int, default=0)
    parser.add_argument(
        "--prompt-languages", default=",".join(PROMPT_LANGUAGES)
    )
    parser.add_argument("--reference-languages", default="ru,ja")
    parser.add_argument(
        "--languages",
        default=",".join(REVIEWED_LANGUAGES),
        help="Comma-separated reviewed language columns",
    )
    args = parser.parse_args()

    encoding = tiktoken.get_encoding(args.tokenizer)
    languages = [language.strip() for language in args.languages.split(",")]
    ratios = {language: [] for language in languages}
    source_token_counts: list[int] = []
    rows: list[dict[str, str]] = []

    with args.csv.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(row)
            source_tokens = len(encoding.encode(row.get(args.base_language, "")))
            if not source_tokens:
                continue
            if len(source_token_counts) < args.benchmark_rows:
                source_token_counts.append(source_tokens)
            for language in languages:
                translation = row.get(language, "").strip()
                if translation:
                    ratios[language].append(
                        len(encoding.encode(translation)) / source_tokens
                    )

    print("| Language | Rows | Median | p90 |")
    print("|---|---:|---:|---:|")
    measured_p90: dict[str, float] = {}
    for language, values in ratios.items():
        p90 = statistics.quantiles(values, n=10, method="inclusive")[8]
        measured_p90[language] = round_up(p90)
        print(
            f"| `{language}` | {len(values)} | "
            f"{round_up(statistics.median(values)):.2f} | {round_up(p90):.2f} |"
        )

    if source_token_counts:
        print("\n| Languages | batch_size | Requests | Planner ms/run |")
        print("|---:|---:|---:|---:|")
        for language_count, ratio in (
            (1, measured_p90["es"]),
            (3, measured_p90["ru"]),
        ):
            for batch_size in (10, 20, 50):
                started = time.perf_counter()
                for _ in range(args.benchmark_iterations):
                    requests = count_batches(
                        source_token_counts,
                        batch_size,
                        language_count,
                        ratio,
                        args.budget,
                    )
                elapsed_ms = (
                    (time.perf_counter() - started)
                    * 1000
                    / args.benchmark_iterations
                )
                print(
                    f"| {language_count} | {batch_size} | {requests} | "
                    f"{elapsed_ms:.3f} |"
                )

    if args.prompt_rows:
        if args.prompt_project is None or args.glossary is None:
            parser.error("--prompt-project and --glossary are required with --prompt-rows")
        asyncio.run(
            measure_prompt(
                project_path=args.prompt_project,
                glossary_path=args.glossary,
                rows=rows,
                row_count=args.prompt_rows,
                dst_languages=args.prompt_languages.split(","),
                reference_languages=args.reference_languages.split(","),
                encoding=encoding,
            )
        )


if __name__ == "__main__":
    main()

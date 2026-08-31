#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import tiktoken


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

    with args.csv.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
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

    if not source_token_counts:
        return

    print("\n| Languages | batch_size | Requests | Planner ms/run |")
    print("|---:|---:|---:|---:|")
    for language_count, ratio in ((1, measured_p90["es"]), (3, measured_p90["ru"])):
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
                (time.perf_counter() - started) * 1000 / args.benchmark_iterations
            )
            print(
                f"| {language_count} | {batch_size} | {requests} | {elapsed_ms:.3f} |"
            )


if __name__ == "__main__":
    main()

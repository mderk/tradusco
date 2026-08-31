# Batch baselines

Measured on 2026-08-31 from
`/Users/max/Documents/projects/t3/client/.tradusco/booty/translations.csv`.
The twelve reviewed locale columns contain 4,369 non-empty rows each. Tokenizer:
`tiktoken` `cl100k_base`. Ratios are translation tokens divided by English source
tokens and rounded upward to two decimal places. Runtime batching uses p90; an
unknown locale uses the largest measured p90 (`3.34`).

Recalculate the table and the local 150-row batching benchmark with:

```bash
uv run python measure_batch_baselines.py \
  /Users/max/Documents/projects/t3/client/.tradusco/booty/translations.csv \
  --benchmark-rows 150 --benchmark-iterations 10000
```

| Language | Rows | Median | p90 |
|---|---:|---:|---:|
| `fr` | 4369 | 1.48 | 2.00 |
| `ru` | 4369 | 2.10 | 3.34 |
| `it` | 4369 | 1.48 | 2.00 |
| `de` | 4369 | 1.48 | 2.00 |
| `es` | 4369 | 1.42 | 2.00 |
| `es-la` | 4369 | 1.40 | 2.00 |
| `ja` | 4369 | 2.10 | 3.34 |
| `ko` | 4369 | 2.17 | 3.25 |
| `pl` | 4369 | 1.74 | 2.80 |
| `pt-br` | 4369 | 1.40 | 2.00 |
| `pt-pt` | 4369 | 1.42 | 2.09 |
| `zh-cn` | 4369 | 1.73 | 2.67 |

## Batch parameters

The fixed set is the first 150 non-empty English rows. The benchmark runs the
same output-budget rule 10,000 times with an 8,192-token output budget. Three
languages use the maximum p90 of `es`, `ru`, and `ja`. `Planner ms/run` excludes
model latency; `Wait at 1s` is the deterministic inter-request delay used by the
translator.

| Languages | batch_size | Requests | Planner ms/run | Wait at 1s |
|---:|---:|---:|---:|---:|
| 1 | 10 | 15 | 0.036 | 14s |
| 1 | 20 | 8 | 0.037 | 7s |
| 1 | 50 | 3 | 0.044 | 2s |
| 3 | 10 | 15 | 0.039 | 14s |
| 3 | 20 | 8 | 0.046 | 7s |
| 3 | 50 | 4 | 0.048 | 3s |

Keep `batch_size=50` and the one-second delay: planner cost is negligible and 50
minimizes calls on this set. The old 2,048-token input limit is removed; output is
now capped at 8,192 estimated tokens.

## Refusal cost

The same 150 rows were translated to Spanish with Gemini 2.5 Flash, one attempt
per request and no artificial delay. The 10-row run used the same model through
OpenRouter after the direct Gemini free tier reached its request-per-minute limit.

| Provider | batch_size | Requests | Wall time | Content filters | Other failures |
|---|---:|---:|---:|---:|---:|
| Gemini API | 50 | 3 | 34.83s | 0 | 0 |
| Gemini API | 20 | 8 | 57.97s | 0 | 1 rate limit |
| OpenRouter | 10 | 15 | 41.96s | 0 | 0 |

The rate-limited 20-row run completed 140 rows; its failed final request cost 10
phrases, confirmed by 10 `failures.jsonl` records. There were no content-filter
refusals in 26 requests, so the corpus provides no evidence for lowering the
default batch size.

For this 150-row set, always using batches of 10 costs 15 requests. Starting with
three batches of 50 and splitting one failed batch into five batches of 10 costs
eight requests total; splitting two failures costs 13, while splitting all three
costs 18. Split retry therefore wins while fewer than all three large batches
fail. It is not enabled here because no content-filter failure was observed and
retry policy is outside output-aware slicing.

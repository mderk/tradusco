# Batch baselines

> **Corpus note (2026-09-04).** The absolute counts below describe the t3 corpus
> as it was on 31 August: 4,369 rows and a 1,071-entry glossary. That phrase list
> has since been rebuilt and now holds 3,556 rows with 1,133 glossary entries, so
> row counts, the entry count and the "glossary entries: 20" figure are stale as
> measurements even where they remain true as conclusions. The per-language
> expansion ratios and the batch-parameter conclusions do not depend on corpus
> size. Recompute with the commands given below before citing any number here as
> current.

Measured on 2026-08-31 from
`/Users/max/Documents/projects/t3/client/.tradusco/booty/translations.csv`.
The twelve human-reviewed locale columns contain 4,369 non-empty rows each:
`ru`, `fr`, `de`, `es`, `da`, `sv`, `no`, `fi`, `nl`, `ja`, `zh-cn`, and
`zh-tw`. These are protected source/reference data, not regeneration targets.
Tokenizer: `tiktoken` `cl100k_base`. Ratios are translation tokens divided by
English source tokens and rounded upward to two decimal places. These measurements
document expected output growth; runtime batching no longer predicts output size.

Recalculate the table and the local 150-row batching benchmark with:

```bash
uv run python measure_batch_baselines.py \
  /Users/max/Documents/projects/t3/client/.tradusco/booty/translations.csv \
  --benchmark-rows 150 --benchmark-iterations 10000
```

## Envelope input size

Measured on 2026-09-01 with the shipped structured prompt, the first 50 non-empty
rows of the same t3 corpus, the 17 regeneration targets, `ru,ja` reference
columns, the real 1,071-entry t3 glossary, and all available neighbor examples.
Tokenizer: `cl100k_base`.

```bash
uv run python measure_batch_baselines.py \
  /Users/max/Documents/projects/t3/client/.tradusco/booty/translations.csv \
  --prompt-project /Users/max/Documents/projects/t3/client/.tradusco/booty \
  --glossary /Users/max/Documents/projects/t3/client/translation_glossary.json \
  --prompt-rows 50
```

| Rows | Languages | Characters | Input tokens | Glossary entries | Rows with references | Examples |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 17 | 47,412 | 22,666 | 20 | 50 | 75 |

The assembled input is below the runtime default of 65,536 input tokens. The
limit is applied to this complete prompt, including context, glossary, references,
examples, and standard-method output instructions.

| Language | Rows | Median | p90 |
|---|---:|---:|---:|
| `fr` | 4369 | 1.48 | 2.00 |
| `ru` | 4369 | 2.10 | 3.34 |
| `de` | 4369 | 1.48 | 2.00 |
| `es` | 4369 | 1.42 | 2.00 |
| `da` | 4369 | 1.50 | 2.25 |
| `sv` | 4369 | 1.50 | 2.34 |
| `no` | 4369 | 1.45 | 2.00 |
| `fi` | 4369 | 1.78 | 2.67 |
| `nl` | 4369 | 1.48 | 2.00 |
| `ja` | 4369 | 2.10 | 3.34 |
| `zh-cn` | 4369 | 1.73 | 2.67 |
| `zh-tw` | 4369 | 2.00 | 3.00 |

## Batch parameters

The fixed set is the first 150 non-empty English rows. This historical benchmark
runs the former output-estimation rule 10,000 times with an 8,192-token budget.
It is retained so the rejected heuristic can be reproduced. `Planner ms/run`
excludes model latency; `Wait at 1s` is the deterministic inter-request delay.

| Languages | batch_size | Requests | Planner ms/run | Wait at 1s |
|---:|---:|---:|---:|---:|
| 1 | 10 | 15 | 0.036 | 14s |
| 1 | 20 | 8 | 0.037 | 7s |
| 1 | 50 | 3 | 0.044 | 2s |
| 3 | 10 | 15 | 0.039 | 14s |
| 3 | 20 | 8 | 0.046 | 7s |
| 3 | 50 | 4 | 0.048 | 3s |

Keep `batch_size=50` and the one-second delay: planner cost is negligible and 50
minimizes calls on this set.

## Runtime policy

- `batch_size=50` remains the first, cheap boundary.
- `batch_max_input_tokens=65536` measures the fully assembled request and halves
  an oversized batch until it fits. A single oversized phrase is still sent.
- Output size is not estimated or capped locally; the model/provider owns its
  completion limits.
- A final `model_error` halves and retries the batch. Authentication, rate-limit,
  and content-policy failures are recorded without splitting.

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
fail. Runtime adaptive splitting now implements this policy for `model_error`.

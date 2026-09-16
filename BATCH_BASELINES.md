# Batch policy and reproducible measurements

Tradusco has two deterministic input boundaries:

- `batch_size=50` limits the number of source rows in an initial request;
- `batch_max_input_tokens=65536` measures the complete assembled prompt and
  halves an oversized batch until it fits.

The measured input includes the prompt, source rows, phrase context, selected
glossary entries, reference-language values and neighboring examples. A single
oversized phrase is still sent. Tradusco does not predict output size locally;
the provider owns its completion limit.

A final `model_error` halves and retries a batch. Authentication, rate-limit and
content-policy failures are recorded without splitting. This avoids multiplying
requests for failures that smaller batches cannot repair.

## Reproduce on your own project

Do not treat numbers from another product as a capacity guarantee. Measure the
actual project CSV and optional prompt inputs:

```bash
uv run python measure_batch_baselines.py \
  /path/to/project/.tradusco/app/translations.csv \
  --benchmark-rows 150 \
  --benchmark-iterations 10000
```

To measure a complete prompt envelope:

```bash
uv run python measure_batch_baselines.py \
  /path/to/project/.tradusco/app/translations.csv \
  --prompt-project /path/to/project/.tradusco/app \
  --glossary /path/to/project/.tradusco/app/glossary.json \
  --prompt-rows 50
```

Record the date, tokenizer, model, language count, reference languages, glossary
size and command with any published result. Recalculate after material prompt,
context, glossary or corpus changes.

## When to change the defaults

Keep the defaults until a bounded run provides evidence that they are wrong.
Reduce `batch_size` or the input ceiling only when reproducible provider failures
correlate with assembled request size. Increase them only when request count is a
measured bottleneck and the chosen model accepts the larger envelope reliably.

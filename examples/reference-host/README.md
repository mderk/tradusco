# Reference host integration

This directory is a small independent host project. It demonstrates the
recommended boundary: the host owns product data and adapters; Tradusco owns
translation state, decisions, model calls, review and coordination.

```text
reference-host/
  source/messages.json       product strings and evidence
  source/terms.json          deterministic terminology
  catalog.csv                host interchange table
  scripts/extract.py         source -> interchange CSV
  scripts/glossary.py        deterministic glossary provider
  scripts/context-provider.js deterministic context provider
  scripts/deliver.py         interchange CSV -> host artifacts
  scripts/artifact_keys.py   built-artifact coverage
  .tradusco/config.json      integration configuration
```

Set the Tradusco checkout used by the commands:

```bash
TRADUSCO_ROOT=/path/to/tradusco
```

The checked-in `traduscoRoot` works while the example remains inside this
repository. After copying the example, set it to the same checkout.

## Prepare without an API call

From this directory:

```bash
node "$TRADUSCO_ROOT/tools/run.js" \
  --config .tradusco/config.json \
  --skip-translate \
  --skip-delivery

node "$TRADUSCO_ROOT/tools/glossary.js" report --config .tradusco/config.json
node "$TRADUSCO_ROOT/tools/context.js" report --config .tradusco/config.json
```

Preparation extracts the current source, synchronizes `.tradusco/app`, writes
generated glossary terms and applies deterministic row context. It does not
change host translations or call a model.

The sample French values are reviewed host data. On the first run, explicitly
import them after preparation rather than asking sync to guess their origin:

```bash
"$TRADUSCO_ROOT/.venv/bin/python" "$TRADUSCO_ROOT/review_translations.py" \
  back-sync --config .tradusco/config.json

"$TRADUSCO_ROOT/.venv/bin/python" "$TRADUSCO_ROOT/review_translations.py" \
  back-sync --config .tradusco/config.json --write --expect REVISION_FROM_PREVIEW
```

## Inspect and translate

Store `OPENROUTER_API_KEY` in an ignored `.env.tradusco` beside `catalog.csv`.
Preview the intended scope:

```bash
node "$TRADUSCO_ROOT/tools/run.js" \
  --config .tradusco/config.json \
  --langs de \
  --dry-run
```

Run the same command without `--dry-run` to prepare, translate, audit, export and
build `dist/de.json`. French is a reviewed reference locale. It can receive new
missing values, but `protectLangs` prevents explicit regeneration.

The example deliberately keeps source text as the translation-memory key. A host
with duplicate source text requiring different translations needs a different
identity design; changing adapter filenames alone cannot provide it.

Use this directory as a reference rather than a framework. Copy only the
adapters supported by the target project and replace the sample product data,
locale policy, build and artifact coverage with real ones.

# Tradusco Integration Guide

This guide explains how to integrate **Tradusco** into another repository while keeping all translation state
in the target project (recommended).

Tradusco is opinionated about its **internal interchange format**:

- a Tradusco “project” is a directory containing `config.json`, a phrase table (usually `translations.csv`),
  and per-locale `progress.json` files
- your application is responsible for extracting strings and building catalogs
- Tradusco is responsible for translating missing strings and maintaining translation memory (`progress.json`)

## Project directory (what Tradusco expects)

Minimal structure:

```
.tradusco/myproject/
  config.json
  translations.csv
  context.txt                # optional
  fr/progress.json
  es/progress.json
```

### `config.json`

Tradusco reads configuration from `config.json`:

```json
{
  "name": "myproject",
  "sourceFile": "translations.csv",
  "languages": ["en", "fr", "es"],
  "baseLanguage": "en",
  "keyColumn": "en"
}
```

Notes:

- **`keyColumn`** is the column that uniquely identifies a phrase. In “phrase-table” workflows it’s usually the
  same as `baseLanguage` (e.g. English text).
- You may include metadata columns in `translations.csv` (see below).

### `translations.csv`

`translations.csv` is a table with:

- one **base** column (e.g. `en`)
- one column per target locale (e.g. `fr`, `es`)
- optional context/metadata columns:
  - `context` (phrase-specific notes)
  - `context_<lang>` (language-specific phrase notes)

Tradusco translates only rows where the destination column is empty (unless `--regenerate` is used).

### `progress.json`

`<lang>/progress.json` is a translation memory mapping:

```json
{
  "Hello": "Bonjour",
  "Goodbye": "Au revoir"
}
```

It is used for caching and can be applied back into other formats.

## Recommended workflows

### Workflow A: “CSV-first” (you already have a CSV phrase table)

1) Create a project dir (either manually or via `create_project.py`)
2) Run `translate.py` for each target locale
3) Consume results from `translations.csv` or `progress.json`

### Workflow B: gettext `.po` (extract → translate → apply)

This is the most common “end-to-end” workflow for gettext/Lingui-style projects.

**Step 1 — extract base msgids into a CSV**

```bash
python extract_translations_csv.py \
  --po-dir locale_src/en \
  --out-csv locale_src/translations.csv \
  --base-col en
```

**Step 2 — sync into a Tradusco project dir**

```bash
python sync_project_from_csv.py \
  --project-dir .tradusco/myproject \
  --source-csv locale_src/translations.csv \
  --base-col en
```

This writes/updates:

- `.tradusco/myproject/config.json`
- `.tradusco/myproject/translations.csv` (prefilled from progress caches when present)

Useful options:

- `--ignore-columns "context,notes"`: if your CSV contains non-locale metadata columns
- `--context-col <name>`: if your context column isn’t named `context`
- `--no-sanitize-progress`: disable mojibake migration + quarantine (enabled by default)
- `--bootstrap-from <dir>`: seed empty `progress.json` files from another cache

**Step 3 — translate missing strings**

```bash
python translate.py \
  -p .tradusco/myproject \
  -l fr \
  -m google/gemini-2.5-flash \
  --method auto
```

**Step 4 — apply progress into PO files**

```bash
python apply_progress_to_po.py \
  --lang fr \
  --project-dir .tradusco/myproject \
  --po-dir locale_src/fr
```

Useful options:

- `--force`: overwrite already-translated entries (default fills only missing/fuzzy)
- `--no-validate-placeholders`: disable placeholder/tag validation when applying

**Step 5 — validate**

```bash
python po_status.py --lang fr --po-dir locale_src/fr --fail
```

Optional: stable diffs

```bash
python sort_po.py locale_src/fr
```

### Orchestrating multiple locales (full loop)

Tradusco intentionally does not ship a project-specific “one button” runner. The recommended pattern is:
keep orchestration in the target repo (so it can run extraction/build commands and manage paths), and call
Tradusco scripts for the translation parts.

Example (gettext/PO):

```bash
# one-time extraction + sync
python extract_translations_csv.py --po-dir locale_src/en --out-csv locale_src/translations.csv --base-col en
python sync_project_from_csv.py --project-dir .tradusco/myproject --source-csv locale_src/translations.csv --base-col en

# per-locale translate + apply + validate
for lang in fr es de; do
  python translate.py -p .tradusco/myproject -l "$lang" -m google/gemini-2.5-flash --method auto
  python apply_progress_to_po.py --lang "$lang" --project-dir .tradusco/myproject --po-dir "locale_src/$lang"
  python po_status.py --lang "$lang" --po-dir "locale_src/$lang" --fail
done

# build catalogs in your app (project-specific)
# e.g. yarn lingui:build / npm run i18n:compile / etc
```

### Workflow C: “Bring your own extractor/applier”

If your project doesn’t use gettext/PO, you can still use Tradusco by treating the CSV as the interchange format:

1) **Export** phrases from your app into a CSV with an `en` (or other base) column and one column per locale.
2) Run `sync_project_from_csv.py` to create/update `.tradusco/<name>/translations.csv` + `config.json`.
3) Run `translate.py` to fill missing translations.
4) **Import** translations back into your app using `progress.json` (or by reading the updated CSV).

## Models & environment variables

Tradusco supports multiple drivers (Gemini, OpenAI, Grok) and OpenRouter.

Common environment variables:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `GROK_API_KEY`
- `OPENROUTER_API_KEY`
- `TRADUSCO_DEBUG=true` (verbose logs)

### OpenRouter model ids

If `OPENROUTER_API_KEY` is set, you can pass a raw OpenRouter model id containing `/`:

```bash
python translate.py -p .tradusco/myproject -l fr -m google/gemini-2.5-flash
```

## Common pitfalls and how Tradusco helps

- **Placeholder/tag mismatches**: Tradusco validates curly placeholders (`{name}`) and Lingui numeric tags (`<0>...</0>`)
  before saving translations.
- **Mojibake keys**: `sync_project_from_csv.py` can migrate common UTF-8-as-Latin1 mojibake keys for current phrases.
- **Whitespace variants**: `apply_progress_to_po.py` includes conservative lookup fallbacks for msgids with trailing spaces
  or “space before newline” sequences.

## Advanced: integrate as a library (custom storage)

For non-filesystem workflows, implement a custom `StorageAdapter` (see `lib/storage/base.py`) and use
`TranslationProject` directly. This allows storing translations/progress in a DB or another system while
reusing the same translation logic.


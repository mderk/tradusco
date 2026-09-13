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
- optional **stable key** column (e.g. `id`) if your application uses string IDs like `CAR_COMMON`
- optional context/metadata columns:
  - `context` (phrase-specific notes)
  - `context_<lang>` (language-specific phrase notes)

Tradusco translates only rows where the destination column is empty (unless `--regenerate` is used).

Notes:

- Column names are **case-sensitive**. `baseLanguage` and `--lang` must match the CSV headers exactly (recommended: lowercase BCP-47-ish codes like `en`, `fr`, `pt-BR`).
- `translate.py` currently uses the **base-language text** as the key for `progress.json` (translation memory). If you keep an `id` column, it will be preserved in the CSV, but it is not used as the translation-memory key.
  - **Base strings should be unique within a project.** If you have the same English text under multiple IDs but need different translations, Tradusco cannot reliably disambiguate them today because both caching and updates are keyed by the base text. Make the base strings distinct (or update the code to key by `id`).

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

Example “keyed” source CSV (IDs + English):

```csv
id,en,fr,es,ru,context
CAR_COMMON,Common Car Box,,,,UI label
BTN_SAVE,Save,,,,Button label
ERR_REQUIRED,Field is required,,,,Validation error
WELCOME_USER,"Welcome, {name}!",,,,Keep `{name}` placeholder
RICH_TEXT,"<0>Learn more</0>",,,,Keep Lingui rich-text tags
```

Create a Tradusco project from it:

```bash
python create_project.py -p .tradusco/myproject -c path/to/translations.csv -b en -k id -i context
```

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

`--lang` takes a comma-separated list, and passing several targets at once is the
cheaper path: the phrases, the context and the glossary are assembled and sent
once for all of them rather than repeated per language. The steps that follow
(apply, validate) are per-locale, so the loop below still applies to them.

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

The generic workflow runner reads project commands and paths from
`tradusco.config.json`:

```bash
node tools/run.js --config /path/to/project/tradusco.config.json
```

It acquires `<projectDir>/.run.lock`, runs configured extraction, synchronizes the
source CSV, prepares glossary and context, and translates missing cells. Each
stage has a `--skip-*` flag; `--dry-run` prints the planned commands. Model calls
have a configurable `translate.requestTimeout` (120 seconds by default).

The equivalent manual pattern remains useful for integrations that need custom
delivery steps:

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

## Auditing completeness & correctness (recommended)

When integrating Tradusco into another repo, it’s useful to have a deterministic “sanity check” step
that catches:

- missing destination cells in `translations.csv`
- invalid JSON-like artifacts accidentally saved as translations (e.g. `{` or `"translations": [`)
- placeholder / Lingui-tag mismatches (`{name}`, `<0>...</0>`)
- progress/cache drift between `translations.csv` and `<lang>/progress.json`

Run:

```bash
python audit_translations.py --project-dir .tradusco/myproject
```

Machine-readable output:

```bash
python audit_translations.py --project-dir .tradusco/myproject --json
```

Fail CI if issues exist:

```bash
python audit_translations.py --project-dir .tradusco/myproject --fail
```

## Translating all locales (parallel)

Tradusco ships a small helper runner that reads `languages[]` from `config.json` and runs `translate.py`
for each locale with a concurrency limit.

Example (Gemini → Grok fallback, run only locales that still have missing/invalid cells):

```bash
python translate_all.py \
  --project-dir .tradusco/myproject \
  --model google/gemini-2.5-flash \
  --fallback-model x-ai/grok-4.3 \
  --parallel 3 \
  --batch-size 50 \
  --batch-max-input-tokens 65536 \
  --method auto \
  --only-missing
```

Notes:

- The runner prints per-locale logs prefixed with `[<lang>]`.
- During parallel runs, Tradusco uses lockfiles like `.translations.csv.lock` and `.<lang>/.progress.json.lock`
  to avoid races. These files are safe to ignore and can be excluded from VCS.
- Per-locale failure logs live in `<lang>/failures.jsonl` (append-only). Check this file when audit
  reports missing cells after a run.

### Single-locale fallback (`translate.py`)

For one locale, `--fallback-model` retries failed batches and then runs a gap-filling pass:

```bash
python translate.py -p .tradusco/myproject -l fr \
  -m google/gemini-2.5-flash \
  --fallback-model x-ai/grok-4.3 \
  --method auto
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

# Tradusco

A translation workflow for preparing glossary and context, translating with
LLMs, reviewing changes and delivering results back to a host project.

## Installation

1. Clone this repository
2. Install the required packages (recommended: `uv`):

```bash
uv sync
```

3. Set up your API keys in a `.env` file:

```
GEMINI_API_KEY = "your_api_key_here"
OPENAI_API_KEY = "your_api_key_here"
GROK_API_KEY = "your_api_key_here"
OPENROUTER_API_KEY = "your_api_key_here"
```

Notes:

- **OpenRouter**: set `OPENROUTER_API_KEY` and pass any OpenRouter model id containing `/`
  (example: `google/gemini-2.5-flash`). Tradusco will route it through OpenRouter automatically.
- **OpenRouter methods**: for compatibility across providers, the OpenRouter driver defaults to the
  standard prompt-based method. If you need structured output or function calling, prefer a direct
  provider driver that supports it.
- **Debugging**: set `TRADUSCO_DEBUG=true` (or pass `--debug` to `translate.py`) to enable verbose logs.

## Development commands

From the repo root:

```bash
make sync
make lint
make fmt
make test

# real API calls (loads .env if present)
make test-integration
```

## Using Tradusco inside another repository (recommended)

Tradusco does **not** require projects to live inside this repo. You can store translation state
in your application repo (for example under `.tradusco/`) and call Tradusco as an external tool.

Typical layout in your app repo:

```
your-app/
  .tradusco/
    config.json
    myproject/
      config.json
      translations.csv
      glossary.json
      contexts.json
      not_terms.json
      deferred_terms.json
      terms_queue.json
      editorial.json
      ru/progress.json
      fr/progress.json
```

Configure the host commands and providers in `.tradusco/config.json`, then run the
complete workflow from the app repository:

```bash
TRADUSCO_ROOT=/path/to/tradusco
node "$TRADUSCO_ROOT/tools/run.js" --config .tradusco/config.json
```

For the current workflow diagram and a step-by-step integration recipe, including
gettext/PO workflows, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).
It contains the current end-to-end workflow diagram, runner configuration,
initial-state rules, review commands, known limits and delivery rules. The
individual Python scripts documented below are low-level engine commands; they
do not run the glossary, context, review and delivery workflow. See
[GLOSSARY.md](GLOSSARY.md) and
[CONTEXT.md](CONTEXT.md) for the two preparation providers. Agents can drive the
complete cycle with [tradusco-run](skills/tradusco-run/SKILL.md).
For a new host repository, [tradusco-init](skills/tradusco-init/SKILL.md)
scaffolds the config and guides the agent through its project adapters.

## Creating a New Project

You can create a new translation project using the `create_project.py` script. This script sets up the necessary directory structure and configuration files based on a CSV file containing your translations.

This is a convenient “CSV-first” bootstrap. For workflows where your app extracts phrases and you want to keep state
in the app repo, also see `sync_project_from_csv.py` in the “Helper scripts for integrations” section.

### Usage

```bash
python create_project.py --path PROJECT_PATH --csv CSV_PATH --base-lang BASE_LANGUAGE --key KEY_COLUMN
```

Or using the short options:

```bash
python create_project.py -p PROJECT_PATH -c CSV_PATH -b BASE_LANGUAGE -k KEY_COLUMN
```

### Arguments

-   `--path`, `-p` (required): Path where the project will be created (the directory name will be used as the project name)
-   `--csv`, `-c` (required): Path to the CSV file containing translations
-   `--base-lang`, `-b` (required): Base language code (e.g., "en" for English)
-   `--key`, `-k` (required): Column name containing translation keys
-   `--ignore-columns`, `-i` (optional): Comma-separated list of column names to ignore (default: "context")

### Examples

```bash
# Create a new project with all required parameters
python create_project.py --path projects/myproject --csv data/translations.csv --base-lang en --key id

# Same using short options
python create_project.py -p projects/myproject -c data/translations.csv -b en -k id

# Create project in a custom location
python create_project.py -p /path/to/custom/project -c data/translations.csv -b en -k id

# Ignore specific columns
python create_project.py -p projects/myproject -c data/translations.csv -b en -k id -i context,notes,comments
```

### What It Does

1. Creates the project directory at the specified path
2. Reads and validates the CSV file
3. Creates a `config.json` file with project settings
4. Creates subdirectories for each language found in the CSV
5. Copies the source CSV file to the project directory

### CSV File Format

The CSV file should have:

-   A column for translation keys (specified by the `--key` parameter)
-   Language code columns for each supported language
-   Each row contains the translation key and corresponding translations

Example CSV format:

```csv
id,en,fr,es
welcome_message,Welcome,Bienvenue,Bienvenido
goodbye_message,Goodbye,Au revoir,Adiós
```

Notes:

- Column names are **case-sensitive**. `--base-lang` and `translate.py --lang` must match the CSV headers exactly (recommended: lowercase codes like `en`, `fr`, `pt-BR`).
- You can add a `context` column (and optionally `context_<lang>` columns) to guide translations for ambiguous strings.
- `translate.py` currently uses the **base-language text** (e.g. the `en` cell) as the key for `progress.json`. If you keep an `id` column, it will be preserved in the CSV, but it is not used as the translation-memory key.

### Project Structure

The utility works with projects that follow this structure:

-   `[project_dir]/config.json` - Project configuration
-   `[project_dir]/<sourceFile>` - Source and destination translations (often `translations.csv`)
-   `[project_dir]/[language]/progress.json` - Translation progress for each language
-   `[project_dir]/[language]/failures.jsonl` - Append-only log of phrase-level failures (why a string is still missing)

### Project Configuration

Each project should have a `config.json` file with the following structure:

```json
{
    "name": "project_name",
    "sourceFile": "translations.csv",
    "languages": ["en", "ru", "de", "fr", "es", "it", "tr", "zh", "ja"],
    "baseLanguage": "en",
    "keyColumn": "en"
}
```

## Translating Phrases

Run the translator with the following command:

```bash
python translate.py -p PROJECT_PATH -l LANGUAGE_CODE [-m MODEL] [-d DELAY] [-r RETRIES] [-b BATCH_SIZE] [--batch-max-input-tokens MAX_INPUT_TOKENS] [--prompt PROMPT_FILE] [--context CONTEXT] [--context-file CONTEXT_FILE] [--method METHOD] [--list-models]
```

### Arguments

-   `-p, --project`: Path to the project directory (either absolute or relative path)
-   `-l, --lang`: Destination language codes, comma-separated (each must be defined in the project's config.json). Several targets in one run share the phrases, context and glossary of each request instead of repeating them per language
-   `--reference-langs`: Comma-separated reviewed columns sent alongside the source to disambiguate it. They are never translation targets, and a cell that is empty for a phrase is simply omitted from that phrase
-   `-m, --model`: Model to use for translation (default: "gemini")
-   `-d, --delay`: Delay between API calls in seconds (default: 1.0)
-   `-r, --retries`: Maximum number of retries for failed API calls (default: 3)
-   `-b, --batch-size`: Number of phrases to translate in a single API call (default: 50)
-   `--batch-max-input-tokens`: Maximum assembled input tokens for one request, counted over the complete prompt including context, glossary, references and examples (default: 65536). An oversized batch is halved until it fits. Output size is not estimated locally — see `BATCH_BASELINES.md`
-   `--prompt`: Path to a custom translation prompt file
-   `--context`: Translation context as a text string to guide the translation style and tone
-   `--context-file`: Path to a file containing translation context
-   `--method`: Translation method to use: auto (automatic selection based on model capabilities), standard (prompt-based), structured (JSON output), or function (function calling) (default: "auto")
-   `--list-models`: List available models and exit

### Examples

```bash
# Translate project in the "projects/myproject" directory to Russian
python translate.py -p projects/myproject -l ru

# Using an absolute path
python translate.py -p /path/to/my/project -l fr -m openai

# OpenRouter: pass a raw OpenRouter model id (must contain "/")
python translate.py -p /path/to/my/project -l fr -m google/gemini-2.5-flash

# Using a relative path
python translate.py -p ./custom_projects/myproject -l de -b 30

# Use a specific model with custom delay and batch size
python translate.py -p projects/myproject -l fr -m openai -d 2.0 -b 20

# Set both batch size and the assembled-input ceiling
python translate.py -p projects/myproject -l de -b 30 --batch-max-input-tokens 16384

# Several target languages in one run, with two reviewed reference columns
python translate.py -p projects/myproject -l it,pl,uk,ro -b 20 --reference-langs ru,ja

# Use a custom prompt file
python translate.py -p projects/myproject -l de --prompt custom_prompts/my_prompt.txt

# Use structured output method (JSON schema)
python translate.py -p projects/myproject -l es --method structured

# Use function calling method
python translate.py -p projects/myproject -l it --method function

# Use automatic method selection (recommended)
python translate.py -p projects/myproject -l fr --method auto

# List available models
python translate.py --list-models

# Retry failed batches and fill remaining gaps with a fallback model
python translate.py -p projects/myproject -l fr \
  -m google/gemini-2.5-flash \
  --fallback-model x-ai/grok-4.3 \
  --method auto
```

When `--fallback-model` is set, Tradusco:

1. Retries each **failed batch** once with the fallback model (e.g. rate limit, provider block, parse error).
2. Runs a **gap-filling pass** at the end if valid cells are still missing (without `--regenerate`).

Phrase-level failures (batch errors, placeholder mismatches, rejected JSON artifacts) are appended to
`<lang>/failures.jsonl` for debugging and CI triage.

## Helper scripts for integrations (CSV + gettext PO workflows)

These scripts are designed to make Tradusco easy to integrate into other codebases.
They are **framework-agnostic**: your app owns extraction/build; Tradusco owns translation state
and correctness helpers.

### `audit_translations.py` (project audit)

Audit a Tradusco project for:

- missing destination cells in `translations.csv`
- invalid JSON-like artifacts saved as translations (e.g. `{` or `"translations": [`)
- placeholder / Lingui-tag mismatches (`{name}`, `<0>...</0>`)
- drift between `translations.csv` and per-locale `<lang>/progress.json`
- UI labels that grew too long to fit their widget (reported as `long_labels`)

```bash
python audit_translations.py --project-dir .tradusco/myproject
```

Fail CI/build if issues exist:

```bash
python audit_translations.py --project-dir .tradusco/myproject --fail
```

#### Length control

Short source strings are captions on buttons, tabs and badges, so a translation
that is much wider than the source overflows or gets clipped. The check compares
**display width** (CJK and fullwidth characters count as two) after removing
placeholders and markup, and it deliberately ignores prose: anything that ends a
sentence, runs past `maxSourceWords` words, or is wider than `maxSourceWidth` is
exempt. Sources that are abbreviations (`HP`, `EXP`, `Lv.`, plus `abbrevSources`)
are held to a much tighter absolute limit, because the widget that shows them is
tiny. As a side effect the check also catches a translation cell that holds a
completely different string.

Over-long labels are reported but do not fail the run unless you ask:

```bash
python audit_translations.py --project-dir .tradusco/myproject --fail --fail-on-length
```

Tune it in the project's `config.json` (all fields optional):

```json
"lengthCheck": {
  "enabled": true,
  "maxRatio": 1.9,
  "minSlack": 6,
  "abbrevSlack": 2,
  "abbrevSources": ["Lvl", "Atk", "Def"],
  "perLang": {
    "de": { "maxRatio": 2.2 },
    "th": { "enabled": false }
  }
}
```

### `translate_all.py` (translate all locales)

Run `translate.py` for every locale listed in `config.json` (with a concurrency limit).

Two-pass example (Gemini → Grok fallback) that runs only locales with missing/invalid cells:

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

### `extract_translations_csv.py` (PO → translations.csv)

Extract unique `msgid` strings from base-locale `.po` files and merge them into a `translations.csv`.

```bash
python extract_translations_csv.py --po-dir locale_src/en --out-csv locale_src/translations.csv --base-col en
```

Notes:

- Reads `.po` as UTF-8 (prevents mojibake issues).
- Intended for common `msgid`/`msgstr` flows. It does not currently treat plural forms (`msgid_plural`) as separate keys.
- Useful options:
  - `--regenerate`: rebuild CSV from scratch instead of merging
  - `--languages "en,fr,es"`: define columns for a brand new CSV file

### `sync_project_from_csv.py` (source CSV → Tradusco project)

Sync an extracted phrase CSV into a Tradusco project directory. This writes:

- `config.json`
- `translations.csv` (prefilled from `progress.json` caches)

```bash
python sync_project_from_csv.py --project-dir .tradusco/myproject --source-csv locale_src/translations.csv --base-col en
```

Quality/safety features:

- **Progress sanitization (default on)**:
  - migrates common UTF-8-as-Latin1 mojibake keys back to UTF-8 (for current phrases)
  - quarantines placeholder-breaking translations into `progress._quarantine.json`
- Preserves metadata columns:
  - `context` (or `--context-col ...`) and `context_<lang>` columns are kept in the output CSV

Useful options:

- `--no-sanitize-progress`: disable sanitization/quarantine
- `--bootstrap-from <dir>`: seed empty `progress.json` files from another project cache
- `--ignore-columns "context,notes"`: comma-separated non-locale columns to ignore when inferring languages
- `--context-col <name>`: context column name in the source CSV (default: `context`)
- `--dry-run`: print summary without writing files

### `apply_progress_to_po.py` (progress.json → PO files)

Fill empty/fuzzy `msgstr` entries from `progress.json` into your `.po` files:

```bash
python apply_progress_to_po.py --lang fr --project-dir .tradusco/myproject --po-dir locale_src/fr
```

Useful options:

- `--force`: overwrite already-translated entries (default fills only missing/fuzzy)
- `--no-validate-placeholders`: disable placeholder/tag validation when applying

### `po_status.py` (status check)

Report missing translations and placeholder/tag mismatches:

```bash
python po_status.py --lang fr --po-dir locale_src/fr --fail
```

### `sort_po.py` (stable diffs)

Sort `.po` files by `msgid`:

```bash
python sort_po.py locale_src/fr
```

### Custom Prompts

The translator supports custom prompt templates for translations.
Default prompts are stored in the `prompts` directory, but you can provide your own prompt file using the
`--prompt` command-line argument.

Prompt templates use Python's string formatting syntax with the following variables:

-   `{base_language}` - The source language
-   `{dst_language}` - The destination language
-   `{phrases_json}` - The JSON array of phrases to translate
-   `{context}` - Global translation context (if any)
-   `{phrase_contexts}` - Individual phrase contexts (if any)

### Translation Contexts

The translator supports both global and phrase-specific contexts to improve translation accuracy. You can provide context in several ways:

#### Global Context

1. **Command-line argument**:

    ```bash
    python translate.py -p myproject -l es --context "This is a videogame translation with casual tone"
    ```

2. **Context file via command-line**:

    ```bash
    python translate.py -p myproject -l es --context-file path/to/context.txt
    ```

3. **Project-level context file**:
   Create either `context.md` or `context.txt` in your project directory:
    ```
    projects/
      myproject/
        context.md  # or context.txt
        config.json
        translations.csv
    ```

All global context sources are combined if multiple are provided.

#### Phrase-specific Context

You can add context for individual phrases by including a "context" column in your translations CSV file:

```csv
en,es,context
Hello,Hola,"Formal business setting"
Goodbye,,"Casual conversation between friends"
Welcome,,"Greeting at hotel entrance"
```

The context column provides specific instructions or background for translating individual phrases. This is particularly useful when:

-   The same word needs different translations based on context
-   There are cultural nuances to consider
-   The phrase has a specific tone or style requirement
-   Technical terms need specific domain context

#### How Context is Used

1. **Global Context**: Applied to all translations in the batch. Useful for:

    - Setting overall tone (formal/casual)
    - Defining domain (technical/medical/legal)
    - Specifying target audience
    - General cultural considerations

2. **Phrase-specific Context**: Applied only to individual phrases. Useful for:
    - Word sense disambiguation
    - Specific tone requirements
    - Cultural adaptations
    - Technical term clarification

The LLM receives both types of context in a structured format, ensuring accurate and contextually appropriate translations.

### How It Works

1. The utility reads the project configuration and source translations
2. For each phrase in the base language, it checks if a translation already exists
3. If no translation exists, it collects phrases into batches for efficient translation
4. It sends batches of phrases to LLM for translation, reducing API calls
5. The translations are saved to both the CSV file and the language-specific progress.json file
6. Translations are cached to avoid redundant API calls
7. The utility implements rate limiting and retries to handle API quotas

### Placeholder / tag validation

Tradusco validates that translations preserve common runtime placeholders:

- `{name}` / `{count}` style curly placeholders
- Lingui-style numeric rich-text tags like `<0>...</0>`

If a mismatch is detected, the translation is skipped (and `sync_project_from_csv.py` can quarantine
already-saved bad entries).

## Core Classes

### TranslationProject

The main class that handles the translation process. Key methods:

-   `async create(project_name, dst_language, prompt_file=None)`: Factory method to create a new instance
-   `async translate(delay_seconds=1.0, max_retries=3, batch_size=50, model="gemini")`: Translate missing phrases
-   `get_available_models()`: Static method to get a list of available models

### LLM Drivers

The project uses a driver architecture for interacting with different LLM providers:

-   `BaseDriver`: Abstract base class that defines the interface for all LLM drivers
-   `GeminiDriver`, `GrokDriver`, `OpenAIDriver`: Concrete implementations for specific providers
-   `get_driver(model)`: Factory function to create the appropriate driver

### Batch Processing

The utility processes phrases in batches to improve efficiency and reduce API calls. Benefits include:

-   **Reduced API Costs**: Fewer API calls for the same number of translations
-   **Faster Processing**: Translating multiple phrases at once is more efficient
-   **Rate Limit Management**: Better handling of API rate limits
-   **Consistent Format**: All translations use the same JSON-based format, even for single phrases
-   **Improved Handling of Multiline Strings**: JSON encoding preserves line breaks and special characters

You can adjust the batch size with the `-b` or `--batch-size` parameter.

## Testing Your Setup

You can verify that your environment is properly set up by running:

```bash
uv run python test_setup.py
```

This will check:

1. If the required environment variables are set
2. If the required packages are installed
3. If the connection to the LLM APIs works

## Running Tests

The project includes a comprehensive test suite to ensure functionality works as expected. To run the tests using pytest:

```bash
# Run all tests (excluding integration tests)
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run integration tests (will make real API calls; loads .env if present)
./tests/run_integration_tests.sh
```

Integration tests are excluded from normal test runs by default to avoid unnecessary API usage and costs. This is configured in the `pytest.ini` file using markers.

The test suite includes:

1. **TranslationProject Tests**: Tests for the main TranslationProject class functionality
2. **Translation Tool Tests**: Tests for the translation tool and API interaction
3. **File Operation Tests**: Tests for CSV and JSON file handling
4. **Prompt Handling Tests**: Tests for loading and handling prompt templates

If you want to run a specific test file or test, you can use pytest with more specific targeting:

```bash
# Run a specific test file
uv run pytest tests/test_translation_project.py -v

# Run a specific test class
uv run pytest tests/test_translation_project.py::TestTranslationProject -v

# Run a specific test method
uv run pytest tests/test_translation_project.py::TestTranslationProject::test_translate -v

# Run tests matching a specific keyword
uv run pytest -k "translate" -v
```

### Integration Tests

The project also includes integration tests that make real API calls to test the translator against actual LLM services. These tests are **excluded from normal test runs** to avoid unnecessary API usage and costs.

#### Running Integration Tests

```bash
# Use the provided script (recommended)
./tests/run_integration_tests.sh

# Run all integration tests directly (note: does NOT load .env)
uv run pytest -m integration -v

# Run specific integration tests file (note: does NOT load .env)
uv run pytest -m integration tests/test_integration_translation_methods.py -v

# Run a specific integration test (note: does NOT load .env)
uv run pytest -m integration tests/test_integration_translation_methods.py::TestIntegrationTranslationMethods::test_standard_method -v
```

#### What Integration Tests Verify

The integration tests verify that all three translation methods work with real LLM APIs:

1. **Standard Method**: Uses basic prompt formatting and parses the JSON response
2. **Structured Method**: Uses the structured output API for more reliable JSON responses
3. **Function Method**: Uses function calling to guide the response format

The tests also include a comparison test that runs all three methods on the same inputs and displays a side-by-side comparison of the results.

#### Requirements for Integration Tests

To run integration tests, you need:

1. An active Gemini API key in your `.env` file
2. An active OpenRouter API key in your `.env` file (for specific models)
3. Internet connectivity to make API calls

# Integration configuration and CLI reference

Tradusco's workflow runner reads `.tradusco/config.json` in the host project.
Every relative path and every configured command uses the directory containing
that file as its working directory. Commands are argument arrays; shell syntax,
redirection and variable expansion are not interpreted.

## Top-level fields

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `traduscoRoot` | path | this checkout | Tradusco installation containing `translate.py` and `tools/` |
| `projectDir` | path | `project` | Mutable Tradusco state, internal CSV and locale progress |
| `sourceCsv` | path | `translations.csv` | Host interchange CSV produced by extraction and updated by export |
| `baseCol` | string | `en` | Source-language CSV column |
| `locales` | string array | `[]` | Translation target columns; translation and delivery require at least one |
| `envFile` | path | `.env.tradusco` | Credentials loaded for child commands |
| `pythonCommand` | string | Tradusco `.venv/bin/python`, then `python3` | Python executable |
| `extractCommands` | argv array list | `[]` | Host commands that refresh `sourceCsv` |
| `glossarySourceCommand` | argv array | omitted | Deterministic glossary provider; receives `--output PATH` |
| `glossaryFile` | path | `<projectDir>/glossary.json` | Generated and reviewed glossary |
| `contextProviderFile` | path | omitted by the runner | CommonJS context provider described in [CONTEXT.md](CONTEXT.md) |
| `contextsFile` | path | `<projectDir>/contexts.json` | Reviewed manual context |
| `glossaryRejectedFile` | path | `<projectDir>/not_terms.json` | Rejected term candidates |
| `glossaryDeferredFile` | path | `<projectDir>/deferred_terms.json` | Deferred term candidates |
| `glossaryQueueFile` | path | `<projectDir>/terms_queue.json` | Context-to-glossary queue |
| `deliveryCommands` | argv array list | `[]` | Host commands that apply and build exported values |
| `artifactKeysCommand` | argv array | omitted | Prints a JSON array of source keys present in built artifacts |

All mutable Tradusco state should resolve below `.tradusco/`. Host source data,
extractors, providers and build commands remain in the host's normal directories.
The working CSV uses the literal `context` column for row context.

### CSV columns

The interchange CSV has exactly three kinds of columns: `baseCol`, `context`
(plus optional `context_<locale>`), and one column per target locale. There is
no key column: the base-language text is the key. When `locales` is set the
runner passes it to sync, and any other column is an error rather than a new
language — a stray `id` column would otherwise be translated as Indonesian.

### Project context files

Besides the context provider, plain files give the model standing guidance:
`<projectDir>/context.md` (or `.txt`) is prepended to every prompt as global
context, and `<projectDir>/<locale>/context.md` only to that locale's prompts.
Tone, formality and audience belong there; per-string meaning belongs in the
`context` column.

### Length check

Audit flags UI labels whose translation is much wider than the source. Tune or
disable it per project in `<projectDir>/config.json`:

```json
"lengthCheck": {"enabled": false}
```

or `{"maxRatio": 2.2, "perLang": {"de": {"maxRatio": 2.5}}}`. Sync keeps a
hand-written `lengthCheck` section. Disable it for forms and documents whose
layout wraps; keep it for fixed-width game UI.

## Translation fields

The optional `translate` object accepts:

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `model` | string | `gemini` | Driver alias or raw OpenRouter model id containing `/` |
| `method` | `auto`, `standard`, `structured`, `function` | `auto` | Model response protocol |
| `batchSize` | integer | `50` | Initial rows per request |
| `batchMaxInputTokens` | integer | `65536` | Maximum complete assembled prompt |
| `requestTimeout` | seconds | `120` | Timeout passed to translation calls |
| `retries` | integer | `3` | Retry budget passed to the engine |
| `delaySeconds` | number | `1` | Delay between model calls |
| `referenceLangs` | string array | `[]` | Reviewed examples supplied to other target languages |
| `protectLangs` | string array | `[]` | Locales that reject explicit `--regenerate` |

A selected target is removed from `referenceLangs` for that run. A protected
locale may still receive translations for missing source keys. The obsolete
inverse field `regenerateLangs` is rejected.

Use an OpenRouter model id such as `google/gemini-2.5-flash` with
`OPENROUTER_API_KEY`. `auto` uses OpenRouter capability metadata to prefer
structured output, then tool calling, then the standard protocol.

## Credential resolution

The runner starts with its inherited process environment and then loads
`envFile`. Values in the configured file override inherited values of the same
name. The file path is relative to `.tradusco/config.json`; therefore a host-root
file is commonly configured as `"envFile": "../.env.tradusco"`.

Keep credentials out of command output, version control and example files. An
authorization failure must stop the run; do not silently switch providers or
models.
Add the configured `envFile` to the host repository's `.gitignore` before
storing a key there, together with Tradusco's transient files:

```gitignore
.env.tradusco
.tradusco/**/*.lock
.tradusco/**/.*.lock
.tradusco/*/failures.jsonl
```

## Runner flags

```bash
node /path/to/tradusco/tools/run.js --config .tradusco/config.json [flags]
```

| Flag | Meaning |
| --- | --- |
| `--config PATH` | Integration config; default `.tradusco/config.json` |
| `--lang LOCALE`, `--langs LIST` | Restrict translation and delivery locales |
| `--model ID` | Override `translate.model` for this run |
| `--only-keys-file PATH` | JSON array of exact source strings; relative to the config directory |
| `--regenerate` | Re-run selected non-editorial cells; rejected for `protectLangs` |
| `--dry-run` | Print current status and commands without acquiring the lock or executing stages |
| `--skip-extract` | Skip host extraction commands |
| `--skip-sync` | Skip rebuilding working state from source and progress |
| `--skip-glossary` | Skip deterministic glossary preparation |
| `--skip-context` | Skip context preview and guarded application |
| `--skip-translate` | Make no model calls |
| `--skip-audit` | Skip the deterministic audit report |
| `--skip-delivery` | Do not export or invoke host build commands |

`--dry-run` is a command plan against current state. It does not execute
extraction or providers and therefore cannot predict their post-run changes.

## Provider and decision commands

Glossary and context decisions are separate from the ordinary runner:

```bash
node /path/to/tradusco/tools/glossary.js report --config .tradusco/config.json
node /path/to/tradusco/tools/glossary.js next --config .tradusco/config.json
node /path/to/tradusco/tools/context.js report --config .tradusco/config.json
node /path/to/tradusco/tools/context.js next --config .tradusco/config.json
```

Their `submit`, `prepare`, `apply`, `lint`, `reopen` and guarded write forms are
documented in [GLOSSARY.md](GLOSSARY.md) and [CONTEXT.md](CONTEXT.md).

## Minimal example

The complete configuration and all four host boundaries—extraction, glossary,
context and delivery—are implemented in
[`examples/reference-host`](examples/reference-host/README.md).

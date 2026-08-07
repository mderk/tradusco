## Tradusco rework plan (OpenRouter Structured Outputs + Pydantic contracts)

### Goal

Make Tradusco’s OpenRouter path as reliable as the “direct provider” drivers by:

- supporting **Structured Outputs** (`response_format: json_schema`) on OpenRouter when the model supports it
- eliminating “JSON scaffolding” leaks into translations (`{`, `"translations": [`, etc.)
- keeping **`progress.json` as primary state** (no truncation / races)
- adding a clear **fallback strategy** when a model refuses or errors (Gemini → Grok, etc.)
- producing deterministic **audit + failure reporting** artifacts for CI and debugging

### Current pain points (observed)

- **OpenRouter runs were prompt-based JSON**, so non‑strict outputs occasionally leaked JSON fragments into `translations.csv` and `<lang>/progress.json`.
- Parallel writers could overwrite each other’s files (fixed with locks/merges, but we should keep the architecture clean).
- Missing visibility into “why a phrase is still missing” (refusal vs parse vs placeholder mismatch vs network).
- OpenRouter Structured Outputs support exists, but the current OpenRouter driver path disables it.

### Proposed changes (phased)

### Phase 0 — consolidate shared validation helpers

Create a small shared module (example: `lib/translation_validation.py`) and use it everywhere:

- `looks_like_json_artifact(value: str) -> bool`
- `is_valid_translation(value: str) -> bool`
- placeholder/tag extraction + comparison helpers

Use this module in:

- `lib/TranslationTool.py` (response parsing + final validation)
- `lib/TranslationProject.py` (treat invalid cache/CSV values as missing)
- `lib/storage/filesystem.py` (merge logic for CSV and progress)
- `audit_translations.py` and `translate_all.py`

Acceptance:

- no duplicated “artifact detection” implementations across scripts/modules

### Phase 0.5 — define “contracts” as Pydantic models (single source of truth)

Goal: represent LLM I/O as typed contracts so we can:

- generate JSON Schema for Structured Outputs from code (no hand-written schemas scattered around)
- parse + validate responses deterministically (CI-friendly)
- keep constraints that JSON Schema can’t express (placeholders/tags) in one place

Work:

- Create a small contracts module (example: `lib/contracts/translation_contracts.py`) with Pydantic models for:
  - Structured Outputs response: `{ translations: list[str] }`
  - Failure records for `<lang>/failures.jsonl`
- Configure contracts to be strict on shape:
  - prefer `extra="forbid"` / `ConfigDict(extra="forbid")` so schemas emit `additionalProperties: false`
- Generate JSON Schema via `model_json_schema()` and feed it into OpenRouter `response_format.json_schema.schema`.

Notes:

- OpenAI-style `strict=true` supports only a subset of JSON Schema → keep models simple (objects/arrays/scalars/enums).
- Keep “semantic” validation (array length == input length, placeholder/tag preservation, artifact rejection) in our shared validation helpers (Phase 0), not in the JSON Schema.

Acceptance:

- translation response schema is defined once (Pydantic) and reused everywhere (driver + parser + tests)
- schemas include `additionalProperties: false` at the top level

### Phase 1 — OpenRouter Structured Outputs (SO) support

OpenRouter supports OpenAI-style:

- `response_format: { type: "json_object" }`
- `response_format: { type: "json_schema", json_schema: { name, strict, schema } }`
- `tools` / `tool_choice`

OpenRouter model metadata can be fetched via `GET /api/v1/models` and checked via `supported_parameters`.
For example, `google/gemini-2.5-flash` reports support for `response_format`, `structured_outputs`, `tools`.

Work:

- Stop hard-disabling SO/tool calling in the OpenRouter path.
- Prefer provider-native SO when available, with a clear method priority:
  - **SO (`json_schema`)** → **tool calling** → **standard text**
- Implement OpenRouter SO in the OpenRouter/OpenAI-compatible driver using the *confirmed* request shape:
  - `response_format.type = "json_schema"`
  - `response_format.json_schema.name = "translations"` (or similar stable name)
  - `response_format.json_schema.schema = <PydanticModel>.model_json_schema()`
  - `response_format.json_schema.strict = true` where supported
- When sending `response_format` / `tools`, set `provider.require_parameters=true` to avoid routing to providers that can’t honor required parameters.
- In `TranslationProject`, prefer SO for OpenRouter models that advertise it, and fall back to standard if:
  - OpenRouter responds with “not supported” errors
  - schema is rejected
  - response is malformed after retries
  - response is truncated (`finish_reason=length`) or blocked (`content_filter`) when using SDK parsing helpers

Notes:

- Keep a short in-memory cache per process for “model supports SO” to avoid repeated capability checks.
- Optionally add an env override like `TRADUSCO_OPENROUTER_SO=force|auto|off`.
- Doc nuance: OpenAI docs sometimes show the **Responses API** (`text.format`) for JSON schema; OpenRouter uses the OpenAI-compatible **Chat Completions** shape (`response_format`) — do not mix these interfaces.
- Optional last-resort: OpenRouter “response-healing” plugin can reduce invalid JSON in non-streaming mode, but should not be required when `strict=true` is working.

Acceptance:

- OpenRouter + `google/gemini-2.5-flash` uses SO by default (auto mode).
- No JSON scaffolding can be saved even under failure modes.

### Phase 2 — failure reporting (why strings are missing)

Add per-locale failure logs, separate from `progress.json`:

- `<project>/<lang>/failures.jsonl` (append-only)

Each record should include:

- timestamp
- model id
- phrase key (if available) + base phrase
- category: `refusal` | `network_error` | `parse_error` | `placeholder_mismatch` | `invalid_artifact_rejected`
- brief message (and optionally a truncated response snippet)

Acceptance:

- For any still-missing phrases after a run, we can point to a failure category and last error.

### Phase 3 — fallback model support in `translate.py`

Add to `translate.py`:

- `--fallback-model <model>`

Behavior:

- Pass 1 uses `--model` as usual.
- If, after pass 1, there are still missing/invalid cells for that locale, run pass 2 using `--fallback-model`
  without `--regenerate` (so it fills only gaps).

Acceptance:

- Single command can “Gemini → Grok fallback” for one locale.

### Phase 4 — dependency hygiene + upgrades (optional, controlled)

Current deps are behind latest (`langchain`, `langchain-openai`, `openai`, etc.).
Upgrades may improve SO/tool calling stability but are major-version changes.

Plan:

- Create a separate “deps upgrade” PR/branch.
- **Hygiene (low risk, do first)**:
  - remove duplicate env libs (`dotenv` vs `python-dotenv`) → keep `python-dotenv` only
  - remove `pathlib` dependency (stdlib on Python 3.13+)
  - add `tiktoken` explicitly (it’s imported; don’t rely on transitive deps)
- **Safe bumps (stay on current API surface)**:
  - `langchain` 0.3.x (ex: 0.3.20 → 0.3.30)
  - `langchain-openai` 0.3.x (ex: 0.3.7 → 0.3.35)
  - `langchain-google-genai` 2.0.x (ex: 2.0.10 → 2.0.11)
  - `langchain-xai` 0.2.x (ex: 0.2.1 → 0.2.5)
  - `google-generativeai` 0.8.x (ex: 0.8.4 → 0.8.6)
  - dev deps: consider `pytest` 9.x and `pytest-asyncio` 1.x after running the suite
- **Major upgrades (do only if needed)**:
  - `langchain` 1.x + `langchain-openai` 1.x introduce interface/migration changes (follow the LangChain v1 migration guide)
  - `openai` 2.x is a major SDK migration (plan separately; only if we need features/bugfixes not present in 1.x)
- After each step:
  - run `pytest`
  - run one-locale smoke translation on OpenRouter with SO enabled
  - confirm fallback behavior still works (SO → tools → standard)

Acceptance:

- no regressions in `pytest`
- successful OpenRouter SO run for one locale

### Phase 5 — tests

Add tests for:

- artifact detection + “invalid values treated as missing”
- progress merge semantics (no truncation, union merge)
- CSV merge semantics under concurrent writer simulation (where feasible)
- response parsing (including line-based fallback) never accepts `{` / `"translations": [` as a translation

### Phase 6 — documentation and example commands

Document:

- `audit_translations.py`
- `translate_all.py` (parallel multi-locale runner)
- recommended two-pass fallback patterns
- lockfiles (`.translations.csv.lock`, `.<lang>/.progress.json.lock`)
- how to check OpenRouter model capabilities (`/api/v1/models`, `supported_parameters`, and `provider.require_parameters`)
- the Pydantic “contracts” (where they live, how to evolve them safely, and how they map to `response_format.json_schema.schema`)

### Acceptance criteria (overall)

- **No missing cells** after full run + fallback (unless explicit refusals remain)
- **No JSON artifacts** in CSV or progress
- **No progress truncation** even if multiple processes run accidentally
- Audit step (`audit_translations.py --fail`) is green in CI for a fully translated project

### Backwards compatibility / rollout

- Keep `translations.csv` format unchanged.
- Keep `progress.json` as a flat mapping; do not mix error records into it.
- Make SO auto-detected + fallback to standard so existing setups keep working.


# Rework plan: the tooling around the engine

> **Status: open — this is the current plan (2026-09-04).** It continues
> `REWORK_PLAN.md` and `REWORK_PLAN_MULTILANG.md`, both closed. Nothing here has
> been implemented yet; only the documentation fixes listed under "Stale
> documentation and loose ends" are done.

Written 2026-09-04 from a review of the Tradusco integration in the t3 project:
23 scripts in `client/scripts/translation-*.js`, 8 in `client/scripts/tradusco/`,
3 shared libraries in `client/scripts/lib/`, and two agent skills that drive the
glossary and context pipelines.

Two questions at once: which parts of that integration are generic and belong in
Tradusco, and what Tradusco itself needs so they can land there.

Every item with a `file:line` reference was verified against the code. Items
without one are proposals, not findings.

**How to read the numbers.** t3 is the only integration measured so far, so it
supplies the evidence: counts of phrases, languages, glossary entries, batches.
Those numbers show that a problem is real and how large it can get. They are not
the specification. Each item below states the generic requirement first and the
t3 measurement second, and where a t3 convention leaked into the requirement it
is called out as a project adapter rather than built in. Nothing in this plan
should assume gettext, lingui, a particular placeholder syntax, or a game.

## Three layers

The boundary does not run script by script. It runs between layers.

1. **Engine** — prompt, envelopes, batching, model calls, response validation.
   Already in Tradusco.
2. **Workflow** — the round trip between a host project's catalogs and a
   Tradusco project, guarded edits, mechanical checks, the format and resolution
   of glossary and context, the run orchestrator. Tradusco has none of this. It
   is roughly two thirds of the t3 integration by file count, and it moves.
3. **Project map** — what a phrase *is* in this particular product: where it came
   from, who says it, what the canonical term is. Never moves.

**Non-goals.** Tradusco must not learn any of the following, however convenient
it looks while there is one integration to test against: that catalogs are
gettext, that placeholders are lingui's, that domains are file names, that a
`context` string has a schema, or that a project has heroes, screens or quests.
Each of those is a provider the project supplies. The test for any item below is
whether a second project with a different stack could adopt it without editing
Tradusco.

The 2/3 boundary has already been drawn once, just in the wrong repository. The
t3 context skill is split exactly along it:

| File | Contents | Layer |
| --- | --- | --- |
| `scripts/lib/context.js` | `.po` domains, config join, manual overrides, resolution order, queues | 2 |
| `sources.js` | which column of which config means what | 3 |
| `context-ui.js`, `context-apply.js`, `context-report.js` | pipeline and report | 2 |

`lib/context.js` knows nothing about t3: it locates the project root by
`tradusco.config.json` and reads everything from there. The glossary side is the
same — `lib/glossary.js` reads `glossaryFile`, `locales` and `regenerateLangs`
from the config, and the only project-specific part is the script that builds the
`terms` section out of the game's config tables.

That gives the shape of the move: **Tradusco owns the format and the resolution,
the project supplies a provider.**

- **Glossary.** Tradusco: file schema (`terms`/`manual`, modes, `cs`, `near`,
  allowed forms), merge order (manual overrides generated), conformance check,
  candidate pipeline. Project: a script emitting `{term: {lang: canonical}}`.
- **Context.** Tradusco: the `context` column, the manual overrides file, the
  resolution chain (manual entry → source column → domain → rule), the deferred
  and rejected queues, the coverage report. Project: the source map that says
  what each column and domain means.

## Language and packaging

Tradusco is Python. The integration is Node, and that is not an accident: this
layer is catalog-format work, and catalog formats are read and written where the
frontend toolchains live. Whatever a host project uses — gettext, ICU, i18next,
Fluent — its parser, its extractor and its compiler are Node packages, and the
integration has to speak to them. Rewriting the layer in Python buys nothing and
costs every one of those parsers.

Proposal: a `tools/` directory in the Tradusco repository with its own
`package.json`, published alongside the Python package. The contract between the
two languages already exists and is already used by both sides —
`tradusco.config.json` is read today by `preflight.js`, `lib/glossary.js` and
`lib/context.js`. Only preflight and status need a Python entry point, because
they gate `translate.py`.

The alternative — keep the code in each project and standardise only the file
formats and CLI surface in the docs — is cheaper now and pays for itself never.
F3 is the case in point: the same defect exists in Tradusco's extractor and in
the t3 copy of it, was found and fixed once in the copy, and is still live here.

## What moves out of the integration

| Code | From | What it is |
| --- | --- | --- |
| G1 | `tradusco/run.js`, `preflight.js` | step orchestrator with skip flags, and the pre-run gate. Project binding is already in the config |
| G2 | `tradusco/sync-from-catalogs.js` | the return path Tradusco lacks: hand-reviewed catalog edits pushed back into `progress.json` and the CSV |
| G3 | `translation-status.js`, `translation-bakeoff.js` | pre-run state, and model comparison on a throwaway project |
| G4 | `translation-scan.js`, `-markup-check.js`, `-script-check.js` | mechanical detectors, see L1 |
| G5 | `translation-apply.js`, `-patch.js`, `-unify.js` | guarded edits: expected `from`, idempotent, conflicts instead of silent overwrite |
| G6 | `translation-lint.js` | glossary conformance, see L2 |
| G7 | `translation-rekey.js`, `-sort.js` | move translations to keys that changed only in whitespace, order catalogs by appearance |
| G8 | `translation-terms.js` | mining glossary candidates from English strings by frequency and mid-sentence capitalisation |
| G9 | `translation-catalog.js` | read one catalog across all languages at once, for review |
| G10 | `tradusco/validate_po.js`, `export_translation_json.js` | catalog validation, and `progress.json` → catalog export. The traversal and the merge rules are generic, the output shape is a project adapter |
| G11 | `scripts/lib/{csv,po,dotted-args}.js` | CSV, `.po` reading and the domain map, argument parsing |

Stays in the project: building the `terms` section from config tables, the
context source map, reading dialogue in script order, per-language style rules,
and review bookkeeping. Runtime coverage (`verify_runtime_coverage.js`,
`check-compiled-catalogs.js`) is a mixed case — see the transport section: the
gate is generic, only the compiler invocation is not.

One cleanup on the way out: t3 currently has two implementations of context
filling — `scripts/translation-context.js` and the skill's `context-apply.js`.
The first is superseded and should be deleted rather than ported. It is not a
tidiness item: on 6 September the superseded script was the one reached for
first, and because it has no manual layer, the manual context was about to be
hardcoded into it. Two entry points for one step means the wrong one gets used.

## Run observability (O)

**O1. The only progress signal says nothing.**
`lib/TranslationProject.py:494` prints `Progress saved: N`, where N is the total
number of filled cells across the target languages. Under `--regenerate` those
cells are already filled and are overwritten in place, so the number does not
move for hours. Judging the 17-language run of 4 September required diffing
`progress.json` against the previous catalog by hand. Print batch *i* of *n*,
phrases done, elapsed time, and an estimate of what remains.

**O2. Recorded failures never surface.** Recovery itself works better than the
console suggests. A `model_error` batch is split in half and retried recursively
(`TranslationProject.py:601`), a policy measured and justified in
`BATCH_BASELINES.md`; an oversized assembled prompt is split the same way before
sending (`:573`); and a language whose block is missing from an otherwise valid
response does not take the other languages down with it
(`TranslationTool.py:396`). Two `model_error` batches out of 82 in the
4 September run printed to the console and left no trace in any
`failures.jsonl`, which is what recovery by splitting looks like.

What does not work is the reporting around it.

- **The exit code and the console are the only channel.** A phrase abandoned for
  good is written to `<project>/<lang>/failures.jsonl` and never mentioned again.
  Nothing aggregates those files, the run does not fail, and no summary says how
  many phrases came back short.
- **Under `--regenerate` the damage is invisible downstream.** An abandoned cell
  is not empty — it still holds the previous value, so every completeness check
  passes and the run looks finished. In the same run, one batch came back without
  the `ro` block and 20 phrases were logged; all 20 still hold their pre-run
  text, and nothing outside that file says so.

Needed: a non-zero exit code when phrases were abandoned, an end-of-run summary
that reads the failure logs it just wrote, and — under regeneration — a record of
which cells were actually rewritten, since that cannot be reconstructed
afterwards.

**O2a. An isolated language failure is never retried.** Decoupling a missing
language block from the rest of the batch is the right first half, and
`WORKFLOW_NOTES.md` names it as a requirement. The second half is missing: those
phrases are recorded and dropped, not re-asked for that language. One malformed
block costs a whole batch in one locale, permanently, and `--ensure-complete`
will not find them because the cells are not empty.

**O3. What was sent and what came back is never recorded.** This is the gap
underneath O2, C2 and R2 alike: every question about a run ends at the same wall.

What exists today. `lib/failure_reporting.py` builds a per-phrase record —
timestamp, model, method, phrase, category, message truncated to 500 characters —
and `filesystem.py:73` appends it to `<project>/<lang>/failures.jsonl`. It works:
in the 4 September run, one batch came back without the `ro` block and 20 records
landed in `ro/failures.jsonl` with `Missing or invalid translation block`.

What is missing.

- **The request body.** The assembled prompt is never written anywhere. Which
  glossary entries survived the cap of twenty, which references and examples were
  attached, whether the phrase went in with context at all — none of it is
  recoverable after the call. Every question in C2, R2 and W5 is a question about
  the request body.
- **The response body.** Only a 500-character error message survives, so a
  failure like `Expecting ',' delimiter: line 427 column 6 (char 52913)` cannot
  be diagnosed at all — the 52 KB that provoked it is gone. Whether the model was
  truncated, refused, or emitted valid JSON in the wrong shape is unanswerable.
- **Whole-batch failures.** The per-phrase log catches a missing language block,
  but the two model-level failures of that same run printed to stdout
  (`TranslationProject.py:405`, `:412`) and left no record in any
  `failures.jsonl`. The log covers the recoverable case and misses the expensive
  one.
- **Successful calls.** Nothing is recorded when a call succeeds, so token usage,
  cache hits, latency and retry counts are unavailable in aggregate. Model
  comparison is a separate script that builds a throwaway project precisely
  because this does not exist.

Proposed shape: an append-only `runs/<run-id>/batches.jsonl` at the project root
with one record per attempt — run id, batch index, model, method, target
languages, phrase keys, token counts and cache counters from the response,
latency, attempt number, outcome — plus `runs/<run-id>/<batch>.request.txt` and
`.response.txt` holding the bodies verbatim. Bodies are the expensive part, so:
always for a failed attempt, on demand (`--log-bodies`) for successful ones, with
a retention limit. API keys never appear in a request body, but the writer should
still redact by allow-list rather than trust that. The run id belongs in the run
summary so a report can point at its own evidence.

**O4. A stalled call is indistinguishable from a working one.**
`lib/llm/openai/OpenAIDriver.py:23` builds `ChatOpenAI(model=..., api_key=...,
base_url=...)` with no `timeout` and no `max_retries`, so the client defaults
apply: 600 seconds per attempt, retried, with nothing printed while it waits.
Combined with O1, a run that has stopped producing anything looks exactly like a
run that is working — no line moves for up to half an hour per attempt, and the
operator's only recourse is to kill it and guess.

Needed: an explicit `timeout` and `max_retries` on the driver, both settable from
the config, and a log line when either fires naming the batch and the elapsed
time. The same applies to any other driver that wraps a LangChain client. The
value matters less than the fact that it is bounded and announced — the failure
this hides is not a slow model, it is a request that will never return (C5).

## Cost (C)

**C1. Cache behaviour is invisible.** The prompt is already laid out for caching:
instructions and context first, phrases last (`prompts/translation.txt`).
OpenRouter charges roughly ten times less for cache reads than for fresh input.
Nothing counts hits, so every claim about savings is a guess. Printing
`input_cache_read` and `input_cache_write` from the response is enough.

**C2. The batch glossary is truncated by entry count.** `lib/envelope.py`,
`_prompt_glossary`: entries are ranked by frequency and the loop breaks at
twenty, which `GLOSSARY.md` documents. Two things are wrong with it. The unit:
the constraint is tokens, and one entry with many inflections across many target
languages costs more than ten short ones. And the silence: nothing says when the
cap binds, so a term that should have been enforced simply is not, and the
translation that comes back looks like any other.

The cap is not hypothetical. `BATCH_BASELINES.md` measures a 50-row, 17-language
request against the real t3 glossary and reports exactly 20 glossary entries —
the cap bound in the ordinary case, on the corpus the defaults were tuned for.
Budget by tokens, and record what was dropped in the run log.

**C3. The glossary block breaks its own cache.** It sits after the stable prefix
and changes from batch to batch. Ordering phrases by which terms they contain
would keep the block identical across a run of batches.

**C4. Whether large-batch-first still holds at 17 languages is worth
re-measuring — but not re-litigating.** Output-side batch estimation was specced
in `A6_BATCH_TASK.md`, implemented, and then deliberately removed in favour of
input-aware batching plus split-on-failure; `BATCH_BASELINES.md` records both the
policy ("output size is not estimated or capped locally") and the arithmetic
behind it: splitting one failed batch of 50 into five of 10 costs eight requests
against fifteen for always-small, so the policy wins until nearly every large
batch fails. That reasoning stands and this plan does not reopen it.

What the 4 September run adds is a data point the baselines do not cover. They
were measured at up to 17 languages for size, but the failure statistics come
from single-language Spanish runs. At 17 targets, two batches of 82 failed hard
enough to exhaust three attempts before splitting, one of them on malformed JSON
at character 52913 — a response the model did not finish. That is still well
inside the policy's break-even. The open question is whether the failure rate
rises with target count, which would move the break-even; the run log from O3 is
what would answer it, and until then the defaults should not move.

**C5. Batch size counts rows, and the cost of a batch is characters times
languages.** `-b/--batch-size` is a count of phrases, so the same number means a
trivial request on short UI labels and an impossible one on prose. The quantity
that has to stay inside the model's output budget is the sum of the source
lengths in the batch multiplied by the number of target languages, and nothing
in the engine computes it.

Measured in t3. A batch of 50 that happened to contain 20 hero backstories
(median 832 characters, longest 1790) against 27 target languages asks for
roughly 500 000 characters of generation in a single structured response. The
request does not fail — it never returns. The same phrases at `--batch-size 3`
completed in 2623 seconds. This is also where the `parse_error` records in
`<lang>/failures.jsonl` came from: truncated JSON is what a response that ran
out of room looks like when it does come back.

This does not contradict C4. C4's arithmetic — split-on-failure beats
always-small — assumes a failed batch fails quickly and cheaply. A request that
exhausts the output budget does not: it either hangs (O4) or burns the full
generation before returning unparseable text, and the split retries pay it
again. The policy holds for batches that are merely large and breaks for batches
that cannot fit at all.

Needed: size the batch by estimated output volume rather than row count. Sum the
source characters, multiply by the number of target languages and a per-language
expansion factor, and cap against a configured budget; `--batch-size` becomes
the upper bound on rows rather than the rule. Prose-heavy corpora then batch
small and label-heavy ones stay large without the operator having to know which
is which. The run log from O3 supplies the expansion factors per language.

## References, examples, regeneration (R)

**R1. The t3 pre-run gate is stricter than the engine needs.** `lib/envelope.py`,
`build`: `reference` is assembled per phrase and an empty cell is simply omitted.
An incomplete reference language does not bother the engine, yet
`preflight.js:66` refuses to start because of one, which is why t3 grew a
`--no-reference-langs` escape hatch. That hatch exists only because the
integration's own argument parser read `--reference-langs ""` as a flag without a
value and produced a language literally named `true` — a trap worth avoiding in
the ported CLI. When preflight moves, downgrade the completeness demand to a
warning: bootstrapping a project has no complete language by definition, and the
engine already tolerates that.

**R2. During regeneration, examples come from the text being replaced.**
`lib/envelope.py`, `_examples`: neighbouring phrases with the same skeleton are
collected and their values are read from `dst_languages`. Under `--regenerate`
the CSV still holds the previous machine translations, so those go into the
prompt as models to follow. Regeneration would then be anchored to the very text
it is meant to replace. **Measure before fixing.** If confirmed, take examples
only from reviewed languages, or disable them under `--regenerate`.

**R3. Regeneration is whole-language, all or nothing.** There is no way to
re-translate a handful of phrases — for instance the ones that just gained a
context line. A `--only-keys` selection, or selection by changed context, would
do it. Today this requires building a separate project.

**R4. The engine has no notion of a protected language.** Reviewed languages are
protected by the t3 orchestrator and by a `regenerateLangs` list in its config.
One misplaced flag overwrites a month of human review. This is a property of the
project and belongs in the Tradusco project config, with a hard refusal.

**R5. Dead keys accumulate and nothing prunes them.** A phrase removed from the
source no longer appears in the phrase list, but its entry stays in
`progress.json` forever. Keeping it is the right default — a string can come
back, and its translation is worth money — but there is no counterpart: no
report of what is dead, no supported way to prune it, and no distinction between
"never translated" and "no longer needed" in any status output. Measured in t3:
5011 keys in each language's `progress.json` against 3556 live phrases, so 1509
dead entries per language across 29 languages, carried and re-saved on every
run.

## Checks and lints (L)

Two properties are checked during a run today: placeholders and lingui tags
(`lib/utils.py`, `placeholders_match` — see F4 for how narrowly that is defined),
and whether JSON scaffolding leaked into the output (`is_valid_translation`).
Everything else lives in the t3 integration, and all of it is generic apart from
the choice of markup delimiters.

`WORKFLOW_NOTES.md` already lists this ground and states the principle this
section only restates: a lint should be a gate that returns the string to the
queue, not a report. It also names the two most productive checks — residual
source language, and a translation identical to its source, which is what a model
emits when it is unsure. L1 to L5 below are that list, plus what a year of
running it added.

**L1. Mechanical detectors belong in the engine.** Translation identical to the
source, Latin text inside a non-Latin locale, the wrong script for the locale
(Cyrillic outside Russian, kana in European languages), two different sources
sharing one translated name, extreme length ratios, junk such as NBSP and double
spaces. Further, from `WORKFLOW_NOTES.md` and not implemented anywhere: the
argument name is syntactically legal for the runtime format (for ICU, no dot —
see the transport section), a leading or trailing marker in the source is present
in the translation, the count of line breaks and paragraph separators matches,
numbers and percentages survive (`95%` in, `95%` out), terminal punctuation keeps
its kind, and per-language typography holds (`«»` against `„“` against `「」`,
dash conventions).

None of these know anything about a game. Two need a project adapter: inline
markup delimiters, and the runtime format that decides which argument names are
legal. They belong in `audit_translations.py` — but they are worth more inside
the run, where a failing cell can be re-asked within the same batch while its
context is still paid for.

**L2. Nothing verifies what the glossary is for.** Entries go into the prompt,
but no check confirms the term actually reached the translation. The t3
conformance check does exactly this, with a list of acceptable inflections, and
it is the only signal that an entry did anything. Without it a bad entry is
invisible until a human reads the catalog.

**L3. Lints have no baseline.** The output is correctly shaped as "places to
read" rather than a verdict, but with no recorded baseline every run reprints the
same accepted places. Recording what was accepted leaves only new noise visible.

**L4. There are no per-language style rules.** Informal address in Russian,
politeness level in Japanese, Du versus Sie in German — every project reinvents
this. A declaration per locale in the project config, plus a hook for a project
script, would cover most of it.

**L5. Lint results feed back nowhere.** A failing cell can be re-asked with the
finding itself as an instruction, which is an order of magnitude cheaper than a
round trip through a human. Today the human round trip is the only path.

**L6. Nothing reports glossary entries that never fire.** An entry that matches
nothing is indistinguishable from an entry that works, so a glossary can be
largely inert without anyone noticing. Two things are missing: a coverage report
saying how many phrases each entry matched over a run, and an explicit choice of
default mode wherever entries are generated, since `exact` matches only a phrase
equal to the term and therefore never applies inside a sentence. Measured in t3:
934 of 1027 generated entries sit in `exact` mode, so 91% of the generated
glossary does nothing for any phrase longer than the term itself. It was found by
reading translations, not by a tool.

**L7. A name in the right script can still be in the wrong form.** Substituting a
canonical name into a sentence is not the same as translating it. The engine can
check the script; only grammar tells whether the form is right, and the check for
that is cheap: count how many occurrences of a name stand in the bare dictionary
form, and flag a bare form directly after a preposition that governs an oblique
case. Measured in t3 on a corrective pass: Ukrainian came back with 2.4% of name
occurrences inflected, which for a language that declines everything means the
names were pasted in, not written. Greek needed a narrower rule, because it
leaves foreign names undeclined and carries the case on the article, so only
names it treats as its own were wrong — 28 occurrences against 468. Thai,
Bulgarian and Arabic needed nothing, since none of them inflect the name itself.
The rule that generalises: the check is per-language and belongs with the
per-language style rules of L4, and the preposition list is the only data it
needs. What does not generalise, and must stay out of the engine, is the
correction itself.

**L8. A string that quotes another string must quote its translation.** UI text
routinely names other controls: *Press "Check status", then enable "Open payments
in the Steam app" in Settings*. Each string is translated on its own, so the
translator never sees the button it is naming, and the instruction ends up
pointing at a label that does not exist on screen. The check is mechanical: take
the quoted spans of the source key, keep those that are themselves source keys of
UI scope, and require the target to contain that key's translation. Quote
characters must not be compared — locales substitute their own («…», „…“, 「…」) —
and prose must be excluded, since quotation marks there are speech, not a
reference.

Measured in t3 the day the string was added: three of the twenty-seven locales
translated the quoted button independently of the button (bg, it, nl), on the
first run, with the glossary already pinning that label. Older damage of the same
kind was sitting unnoticed in sixteen places, one of them in nine locales at
once. The data the check needs beyond the catalogs is which keys are UI — the
same scope that F1 puts in the domain column.

**L9. The fixes layer can silently contradict the glossary.** Where a project has
both a pinned-override layer applied at export and a glossary canon, nothing
compares them, and the override wins without a word. In t3 the export config
pinned `Oblivion` to Latin for every locale but three; the glossary had since
grown non-Latin canon for five more. Every export quietly rewrote those five back
to Latin, so a hand fix in the catalog survived exactly until the next export, and
the only reason it surfaced is that a Latin-script lint was run afterwards. A
term present in both places with different values is always a defect in one of
them, and the comparison is a dictionary lookup at export time.

## Format and cohesion (F)

**F1. `scope` and `except` are dead fields, and the fix is already specified.**
`GLOSSARY.md:78` explains why they are dead — Tradusco does not know where a
phrase came from, so it cannot apply a scope — and then names the remedy in one
sentence: deterministic scope and catalog filtering require an explicit row field
supplied by the source project. `WORKFLOW_NOTES.md` repeats it. Nothing here is a
new idea; what is missing is that nobody has added the column. It is a domain
column in `translations.csv` — an opaque label saying which part of the product a
row belongs to. Where the label comes from is the project's
business: a catalog file name, a table, a key prefix, a directory. Tradusco only
has to carry it, expose it to the glossary rules and to context resolution, and
report on it. That revives three things at once: term scope, per-domain
exceptions, and domain-derived context. It does not replace the manual context
file, which exists for strings whose meaning is not implied by where they live.
This is the most connected change on the list, and the one that most needs to
stay opaque — the moment Tradusco interprets domain names, it acquires a
project's taxonomy.

**F2. The glossary file is not schema-checked.** A typo in a mode silently
becomes `exact` and the entry stops working without saying so. In a 600 KB file
this is not findable by eye — and see L6 for how much rides on the mode.

**F3. `extract_translations_csv.py` loses phrases that changed.**
`_normalize_key` collapses whitespace (`extract_translations_csv.py:12`); the set
of existing keys is built from the normalised form (`:135`) and new phrases are
filtered against it (`:150`). `--collapse-whitespace` defaults to true (`:95`).
A phrase whose only change was line endings never enters the list and becomes
permanently untranslatable. The same defect existed in the t3 extractor and cost
two live strings that sat untranslated for six months.

**F5. Phrase-as-key is the only identity, and it decides more than it looks.**
`progress.json` is keyed by the source string, which `GLOSSARY.md` notes makes
one thing impossible outright: two identical source phrases that must translate
differently in different places cannot be represented at all, no matter how good
the glossary or the context is. `WORKFLOW_NOTES.md` works the general problem
through and lands on stable id plus a hash of the source text stored beside the
translation — gettext's fuzzy flag, computed explicitly — decided per corpus
rather than per project, because config-derived strings usually have ids and UI
strings usually do not. That is the largest open design item in this repository,
it is prerequisite to any honest answer on R5 and on rekeying, and it should be
scoped before anything builds further on the current key.

**F4. The placeholder and tag syntax is hardcoded.** `lib/utils.py` defines
`_CURLY_TOKEN_RE` for `{token}` and `_LINGUI_TAG_RE` for lingui's numbered tags,
and `placeholders_match` — the only validation applied to every cell of every run
— is built on them. A project using ICU plurals, printf-style `%s`, `{{var}}`,
Rails-style `%{var}` or HTML gets either no protection or false rejections, with
no way to say so. This is the one place where Tradusco is already tied to one
project's conventions, and it sits in the hottest path in the codebase. The
patterns belong in the project config, with the current pair as the default.

## Transport to the runtime (T)

A translation that is correct and never reaches the player is indistinguishable
from a bad one, and this class is absent from everything above. `WORKFLOW_NOTES.md`
documents it from a t3 incident: argument names generated from config column
paths (`{am.holy}`) are illegal in ICU, `lingui compile` emitted 783 `invalid
syntax` warnings, **exited 0**, and dropped every affected string from the
compiled catalog. All 29 locales showed English to the player with the numbers
interpolated correctly, which is exactly why nobody noticed.

**T1. A compiler's exit code guarantees nothing.** The gate is a set comparison
of keys before and after each step of the chain — source, base format,
compilation, runtime — failing on any loss. It is fifty lines and it catches the
whole class. Generic: only the command being wrapped is a project adapter.

**T2. Argument names are part of the contract, not a source detail.** When names
are generated upstream they reach both the runtime and the translator's screen.
They must be rewritten at the build boundary (`{am.holy}` → `{am__holy}`) with
the reverse applied in the runtime, which makes the rewrite one shared function
rather than two copies that drift. Any key-set comparison has to know about the
rewrite, or it reports equal numbers of `missing` and `extra` — the signature of
comparing two spellings of the same key.

t3 has both as working prototypes (`check-compiled-catalogs.js`,
`lib/dotted-args.js`), which is where the generic version should be taken from.

## Orchestration (W)

**W1. Multi-language runs work in the engine but not around it.** `translate.py`
accepts `-l fr,de,es` and sends phrases, context and glossary once for all
targets, which is the cheap path. The surrounding steps in the t3 orchestrator
still assume one language and pass the raw string on — producing a path like
`locale_src/fr,de,es`. Export and apply have to be run separately today. If the
whole loop lives in Tradusco this cannot drift.

**W2. There is no plan or dry run.** Before spending money there is no way to
ask: how many batches, how many tokens, what will this cost, which phrases go in
without context or glossary coverage. All the inputs exist.

**W3. Filling gaps works, repairing damage does not.** `run.js --ensure-complete`
escalates the batch size downward (configured, then 20, then 10) and finally
moves to a fallback model on batches of 10, 5 and 1, and
`sync_project_from_csv.py` quarantines translations that break placeholders into
`progress._quarantine.json`. Both find *empty or invalid* cells. Neither finds a
cell that is populated but wrong — the abandoned-under-regeneration case from O2,
or the isolated language failure from O2a. With O3 in place, re-running exactly
the phrases a run failed on becomes a selection over the run log rather than a
search for holes that are not there.

**W4. Long runs have no supported shape.** A full regeneration takes hours. There
is no heartbeat, no way to ask a running job where it is, and no guidance on
detaching. Each project rediscovers this.

**W5. The pre-run gate should cover inputs, not just references.** Preflight
checks reference completeness and hashes the glossary. It does not report how
many rows are about to go to the model with neither context nor a glossary
entry — the one number that predicts rework, and the cheapest thing to print
before spending money. Seen in t3 on 4 September: 14 phrases from a new feature
went into a 17-language run bare, because the context step had not been re-run
after they appeared. The number was computable and nobody computed it.

**W6. The incremental cycle has no entry point.** Everything is shaped for a full
run, but the ordinary event in a live product is different: a handful of new
source strings appear, and they need the same four steps as a full run —
extraction, glossary, context, translation — over a much smaller set. Nothing
supports that shape. Extraction has a command, but it reports nothing about what
it changed, so *which strings are new* has to be recovered by diffing the source
catalog by hand; the glossary and context passes have no way to be told "only
these"; and the translation step then has to be ordered by hand (W7).

Measured in t3: three new UI strings cost roughly twenty operator actions, of
which three were the actual decisions — one glossary entry pair, three context
lines, one run. The rest was rediscovering the state of the pipeline. Needed: one
command that extracts, prints the added and removed keys, runs the glossary and
context passes restricted to the added ones, and hands them to the run in the
right order.

**W7. For new rows the reference languages are missing by construction.**
Preflight refuses to start when a reference locale lacks rows, which is right for
a stale reference and wrong for a new string: nothing has been translated into
anything yet, so every reference is incomplete on exactly those rows. The escape
hatch exists (`--no-reference-langs`), but the order it implies — translate the
reference locales first, then the rest against them — is written down nowhere and
is re-derived from a failed preflight each time. Preflight should separate "the
reference is missing on rows no locale has yet" from "the reference is
incomplete", and the orchestrator should sequence the two phases itself.

## Skills for agents

- **`tradusco-glossary`** — the candidate pipeline: present one candidate with
  all evidence, take a decision, run the conformance check before and after, and
  roll the entry back if it raised noise. Minus the term source, which is
  project-specific.
- **`tradusco-context`** — the same for context: one screen's worth of strings
  per turn, answer by file, validation and write. Minus the source map.
- **`tradusco-run`** — pre-run gate, detached long run, coverage check, export,
  apply, build. Carries the long-run rules from W4.
- **`tradusco-qa`** — every mechanical check in one pass with one report.
- **`tradusco-review`** — already specified in full in `REVIEW_SKILL_SPEC.md`,
  down to the defect categories, the boundaries and the five script roles it
  needs (reader, writer, glossary check, structural lints, progress). Nothing to
  design; what is missing is that four of those five roles now exist as the G3 to
  G6 scripts, so the spec should be re-read against them rather than implemented
  from scratch.
- **`tradusco-init`** — bring up a new project, and check an existing one. See
  below.

### `tradusco-init`

Two commands, one skill. `check` is not only for new projects: it is the thing
that answers "is this repository still wired correctly" after any change, and it
should be runnable on its own.

**`init` — an interview that mostly confirms.** The failure mode of a scaffolding
questionnaire is twenty blind questions. Almost everything here is detectable
from the repository, so the skill detects first and asks only to confirm what it
found or to settle what it genuinely cannot know:

- *Detected:* catalog format and location (`.po` files, an i18n JSON tree, XLIFF,
  a lingui or i18next config), the locale list and the base language, the
  extract and build commands from `package.json` scripts, and the placeholder
  syntax — sampled from actual base-language strings, which settles F4 without
  asking anyone.
- *Asked:* which locales are human-reviewed and must never be regenerated (R4),
  which are references, the model and batch settings, and whether the product has
  a domain notion to put in the domain column (F1) — file names, table names, key
  prefixes, or none.

The interview itself belongs in the host's structured question facility
(`AskUserQuestion` in Claude Code) rather than in a script prompting on stdin: it
renders as a choice list with the detected value offered first, it survives the
agent being interrupted, and it keeps the skill free of terminal input handling.
The script's job is to detect, hand the agent a set of questions with defaults,
and consume the answers as a file — the same shape the other two pipelines use.

Output: a `tradusco.config.json`, and scaffolded project-local providers with one
worked example each rather than empty files — a glossary term source emitting
`{term: {lang: canonical}}`, a context source map with one real domain mapped and
the rest listed as `TODO`, and a `README` naming which file to edit next. The
skill then hands off to `tradusco-glossary` and `tradusco-context`, which are
built to be driven by an agent from exactly that state.

The honest boundary: scaffolding is cheap, and a *good* context source map is not
— it encodes what the product's strings mean, which is the layer-3 knowledge that
never moves. `init` should produce a skeleton that runs and one example that
works, then stop. Generating more would generate guesses.

**`check` — validate a project against the contract.** Config parses and every
path in it exists. The extract command produces a CSV with the expected columns.
The base language has no empty cells. The configured placeholder patterns
actually match the placeholders present in the base strings, and no base string
contains a token no pattern matches — the check that would have caught F4 in any
project that is not t3. The glossary file matches its schema (F2) and its
entries are reachable in some mode (L6). Reference languages exist, protected
languages are listed, context coverage is reported (W5). Dead keys are counted
(R5). Exit code says whether a run is safe to start.

Three constraints learned the hard way in t3, worth carrying over:

- **The agent answers with a file, not a pipe.** Piping into a script from an
  interactive agent session asks the user for confirmation every time. Writing a
  JSON answer file and calling `submit` does not.
- **Data files must not live inside the skill directory.** Writes under the
  agent's configuration directory prompt for permission, and these pipelines
  write in batches. Glossary, contexts and queues live in the project root next
  to `tradusco.config.json`.
- **A skill needs a decision it may not take.** Both pipelines stop and ask when
  a candidate is a naming decision rather than a mechanical one. That is what
  keeps them safe to run unattended.

## Stale documentation and loose ends

Found while checking this plan against the rest of the repository. The first
group was corrected on 2026-09-04 and is recorded here so the fixes are not
redone; the second group is still open.

Convention applied while fixing them: a document that is no longer work to pick
up carries a `> **Status: ...**` blockquote as its first element after the H1, so
an agent reads the status before the content.

### Corrected

- **`README.md` and `INTEGRATION_GUIDE.md` described one language per run.**
  Fixed. `-l, --lang` is now documented as a comma-separated list with the
  shared-request rationale, `--reference-langs` is documented, the
  `--batch-max-tokens` / 2048 entry was corrected to `--batch-max-input-tokens` /
  65536 with the halving behaviour, and the guide's per-locale loop now says which
  steps stay per-locale.
- **`A6_BATCH_TASK.md` read as pending work that was in fact done and reversed.**
  Fixed. It now opens with a **cancelled — do not implement from this document**
  status naming both commits and pointing at the `BATCH_BASELINES.md` runtime
  policy.
- **`A1_MULTILANG_TASK.md`, `A2_A5_ENVELOPE_TASK.md`, `REWORK_PLAN_MULTILANG.md`,
  `REWORK_PLAN.md`** carried no completion state. Fixed: each now opens with a
  done/closed status naming the commits, and `REWORK_PLAN.md` additionally flags
  that it predates multi-language runs.
- **`BATCH_BASELINES.md` measured a corpus that no longer exists in that shape.**
  Marked rather than recomputed: a corpus note now says which numbers are stale as
  measurements, which conclusions do not depend on corpus size, and to recompute
  before citing. Recomputing needs a run and is not blocking.
- **`WORKFLOW_NOTES.md` had three statements overtaken by events.** Fixed in
  place: the glossary now exists and points at `GLOSSARY.md`, the context column
  figure is 3,494 of 3,556, and the context-mining skill question is struck
  through and closed. Its batching section now separates what shipped from the
  one part that did not (marking the specific row a content filter tripped on).
- **t3's `TRADUSCO_WORKFLOW.md` was wrong about export.** Fixed: it now describes
  the four actual layers (base phrases from progress, `translation_fixes.json`,
  `exportFixes`, then preserved non-base keys) and says plainly that a hand edit
  is reverted only when the key is in the current phrase list.

### Still open

- **A one-line defect recorded and left standing:**
  `export_translation_json.js:209` sorts keys with `localeCompare` although
  `baseMsgids` already arrive in the correct order and a `Set` preserves
  insertion order. It is in the G10 move, so it should be fixed on the way, not
  ported.
- ~~**`exportFixes` and `translation_fixes.json` are implemented and unused.**~~
  Closed on 6 September, and the answer was the first branch: the durable layer
  belongs inside the guarded edit. `translation-apply.js` now writes every
  applied edit to the catalog and to `translation_fixes.json` in one step, so an
  edit cannot be left in the perishable half. Carry that into G5 rather than
  treating the two writes as separate operations. What the same day exposed is
  L9: the override layer and the glossary can disagree, and nothing notices.

## Order of work

0. **O4 and C5**, ahead of everything else, because together they are the one
   failure that stops a run rather than degrading it: a batch too large to fit
   in the output budget, waited on by a client with no timeout. O4 is a few
   lines on the driver. C5 is a day at most, and it is what makes a prose-heavy
   corpus runnable at all without hand-picking the batch size.
1. **O3 first, then O1, O2, O2a.** O3 leads because every other item on this list
   is a question about a request or a response, and none of them can be answered
   today. O1 and O2 are about half a day on top of it: the failure records
   already exist, they only need surfacing and an exit code. O2a — retrying a
   language whose block was lost — is the one place in this group where phrases
   are being abandoned for good.
2. **W5, W2** — the pre-run gate reporting context and glossary coverage, and a
   dry run that prices the job. Cheap, and both prevent rework rather than
   detecting it.
2a. **W6, W7** — the incremental cycle and the phase order it implies. Placed
   here because it is the only item on this list that pays back on every ordinary
   week rather than on a big run, and because W5 is most of its report: the gate
   already has to know which rows are new and which lack context. W7 is an
   afternoon; W6 is the command that ties the four steps together.
3. **F4** — placeholder patterns into the config. Small, and it is the one
   coupling to a single project already inside the engine, so it should not
   survive the first item of decoupling work.
4. **F1**, the domain column — unblocks more of the rest than anything else.
5. **L6, L2, L1** with findings fed back into the batch — the only place where
   checks start saving money rather than only time. L6 leads because an inert
   glossary is a wrong answer delivered confidently, and nothing currently
   reports it: in t3 it went unnoticed at 91%.
6. **T1, T2** — the transport gate. Cheap, mechanical, and it covers a failure
   that is invisible by construction: correct translations that never reach the
   product.
7. **G2, G1** — the return path and the orchestrator. Without them any second
   project starts by losing its human review.
8. **F5**, key identity — the largest open design item, and the one that should
   be scoped before more is built on the current key.
9. **Skills** for glossary and context, together with the layer-2 halves of both
   subsystems. Then `tradusco-init`, which is what turns all of the above into a
   second project that costs a day instead of a month — and whose `check` half is
   worth having before that, since it is the only thing that tests whether the
   decoupling actually held.

Separately and before all of it: **measure R2.** If examples really do anchor
regeneration to the old text, that changes the quality of everything translated
from scratch, and it needs to be known before the next large run. This one does
not wait on O3: `EnvelopeBuilder` is pure, so building an envelope from the
current CSV with `--regenerate` semantics and reading what comes out answers it
without calling a model.

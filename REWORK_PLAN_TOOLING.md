# Rework plan: the tooling around the engine

> **Status: open — this is the current plan (2026-09-04, reviewed 2026-09-07).**
> It continues `REWORK_PLAN.md` and `REWORK_PLAN_MULTILANG.md`, both closed.
> Nothing here has been implemented yet; only the documentation fixes listed
> under "Stale documentation and loose ends" are done. An outside review on
> 7 September checked the items against the code: its findings are folded into
> the text, and the items it demoted are now marked as proposals.

Written 2026-09-04 from a review of the Tradusco integration in the t3 project:
23 scripts in `client/scripts/translation-*.js`, 8 in `client/scripts/tradusco/`,
3 shared libraries in `client/scripts/lib/`, and two agent skills that drive the
glossary and context pipelines. These are the original inventory counts, not a
current recount.

Two questions at once: which parts of that integration are generic and belong in
Tradusco, and what Tradusco itself needs so they can land there.

**Product goal (confirmed by the owner, 2026-09-07).** Tradusco is to be a
standalone tool usable in other projects. Recent t3 translation work supplies
new reusable patterns and workflows, including deterministic operations, that
should enrich the tool itself. Project-dependent tools remain a supported and
necessary layer: the goal is to transfer common behaviour while retaining that
boundary, not to absorb the host project's knowledge or to keep Tradusco a
t3-only utility. This goal does not approve individual design proposals or
authorise implementation; the current phase is document agreement.

**First-upgrade scope.** Implement only the reusable parts of the proven t3
workflow listed as G1–G11: glossary and context preparation, their use in model
input, deterministic checks and reports, guarded editorial sync, orchestration,
resume and export. Move product facts and catalogue-specific access behind project
providers. Preserve the current engine, source-as-key storage and batching policy
unless a transferred operation cannot work without a small compatible change.

The first upgrade does not include a new identity store, owner/domain identity
modes, legacy identity migration, a general transaction system, concurrent-writer
support, new semantic or language heuristics, cache/batch experiments, an init
interview or parity across multiple new adapters. Those are later work justified
only by a real target project. They must not block the G1–G11 transfer and are
tracked in [`BACKLOG_TOOLING.md`](BACKLOG_TOOLING.md).

Every item with a `file:line` reference was verified against the code. Items
without one are proposals, not findings, and are labelled as such.

The counts were re-taken on 7 September and several had moved or had been read
wrongly the first time; where that happened the item says so rather than quietly
carrying the new number. A figure here is evidence that a problem exists and how
large it can get, and it goes stale the moment the corpus changes — recompute
before citing one as current.

**How to read the numbers.** t3 is the only integration measured so far, so it
supplies the evidence: counts of phrases, languages, glossary entries, batches.
Those numbers show that a problem is real and how large it can get. They are not
the specification. Each item below states the generic requirement first and the
t3 measurement second, and where a t3 convention leaked into the requirement it
is called out as a project adapter rather than built in. Nothing in this plan
should assume gettext, lingui, a particular placeholder syntax, or a game.

## End-to-end workflow: the organising specification

**Status: assembled for whole-process review, not implementation.** This section
organises the agreed behaviour into one process. The ordinary translation,
editorial-priority, partial-export and input-readiness rules were agreed with the
owner; connecting details and explicitly open cases below remain proposals.
Review this process first. The O/C/R/L/F/T/W/G inventory afterwards supplies
evidence and implementation concerns, not a second independent workflow.

The objective is a reusable round trip: connect a project, recognise its changes,
prepare translation inputs, translate the selected cells, validate and persist
them, deliver them to the host, and incorporate subsequent editorial work.
Tradusco owns common operations and their outcomes. The project supplies source
meaning, the current source-key mapping and host integration. No step assumes a game,
gettext, a specific agent host or Node in every project.

### Reading the t3 evidence

The t3 working checkout is not a stable evidence location: it is often on
`master`, which lacks `scripts/tradusco`, and so does `common/main`. The
integration lives at the tip of `steam/dev` (`143113d59d` at this review),
54 files including both agent skills, `translation_glossary.json` and
`translation_contexts.json`. Read it without switching branches:

```sh
git -C /Users/max/Documents/projects/t3/client show 143113d59d5a2c7c43ef07a1a32bd75440826d1f:scripts/tradusco/run.js
git -C /Users/max/Documents/projects/t3/client show 143113d59d5a2c7c43ef07a1a32bd75440826d1f:.claude/skills/t3-translation-context/sources.js
```

Use this single pinned snapshot for reproducible comparisons across files.
`steam/dev` can be used to inspect later work, but record its resolved commit
before measuring. Earlier file/line citations may predate this snapshot.

### Ordinary path

| Step | Inputs and Tradusco action | Project responsibility | Output and next step |
| --- | --- | --- | --- |
| P0. Connect | Read configuration and declared sources/formats; validate the integration and reuse existing project state. | Supply source/catalogue access, locales, format support and glossary/context providers. | Continue to P1 with the current source-as-key project. |
| P1. Reconcile and select | Compare source keys, guidance changes, protected edits and host changes. Select new, changed and explicitly regenerated work. | Supply the current source snapshot, affected scope and regeneration policy. | Explicit work selection. Protected values remain protected; no relevant change means no model work. |
| P2. Prepare and decide inputs | Gather candidates/evidence through deterministic providers and optional glossary/context skills. Resolve existing decisions; where needed, prepare and accept new naming/context decisions before translation, within granted authority. Maintain accepted/deferred/rejected decisions. Then assemble guidance, references and diagnostics. | Supply product facts, canonical/style decisions, decision authority and required/optional input policy. | Ready inputs for P3 or specifically deferred preparation. Missing optional guidance alone does not force approval. Read-only inspection shows existing inputs and pending decisions without running inference or persisting changes. |
| P3. Translate | Translate eligible selected cells from the source, sharing requests across target locales where applicable. Use current batching policy and declared models, fallback and bounded retry settings. References guide meaning and carry honest provenance. | Select models, permitted fallback, targets, optional reference-first ordering and resource limits. | Candidate results and attempt outcomes to P4. No automatic regeneration of editorial text for the same source; changed source is translated without mandatory review. |
| P4. Validate and persist | Check response/format, repair technical failures within budget and report heuristics. Include cross-locale discrepancies where a supported check can detect them; semantic comparison is a separate inference operation, never an assumed free deterministic check. Reconcile intervening changes and record persisted outcomes. | Supply format/check policy and authorised quality-comparison scope. | Ready cells to P5 and unresolved work. Build on existing ensure-complete and placeholder quarantine, closing their outcome/selection gaps rather than replacing them. Majority agreement is not semantic proof. |
| P5. Deliver | Reconcile pending host edits, export ready valid cells without erasing unrelated/unresolved values, invoke requested build steps and compare expected identities with output artefacts. Record translation completion separately from delivery. | Supply writer, compiler/build command, artefact reader and identity normalisation; own fallback and release policy. | Complete or partial delivery report. Retry failed delivery from persisted translations, without retranslating them. Export to project files does not itself publish a release. |
| P6. Review and improve, when requested | Present related records in project order across locales with source/context. Accept scoped editorial decisions, apply guarded corrections through the common write operation and preserve review provenance. Suggest reusable glossary/context/check improvements separately from individual corrections. | Supply grouping/order, usage or speaker facts, reviewer authority and editorial decisions. | Corrected cells return through P4's validation/persistence and P5's delivery, without a translation call merely to apply a decision. Accepted reusable guidance becomes input for P2 in a later cycle. |
| P7. Continue | On a later invocation, reconcile current state and resume unresolved translation or undelivered persisted results. New source/host changes return to P1; optional review returns to P6. | Choose the next scope or resume operation. | A new bounded invocation with preserved history; completed work is not repeated merely because another cell or stage failed. |

The automatic path is P1 → P2 → P3 → P4 → P5. P0 is initial setup or an
integration change. P6 is an independently invoked editorial path, not a gate
between every translation and export. Already persisted work can go from P1
directly to P5; approved corrections enter P4 without P3. These paths expose the
same deterministic operations to CLI users and agent skills.

**Identity scope for this upgrade.** Keep the current source-as-key convention:
one managed translation per exact source key. Source changes create new work and
old progress remains available as history. Stable-ID storage, owner namespaces,
domain-separated values and legacy identity migration are deferred and do not
block glossary, context or pipeline transfer.

P2 contains a preparation loop: gather evidence → reuse existing decisions and
rules → prepare missing values → resolve decisions within granted authority →
validate and apply authorised changes → return to the remainder. Glossary and
context decisions can occur before the first translation; P6 can feed later
findings into the same loop. Rejected candidates (including “not a term”) are not
proposed again on unchanged evidence. Deferred work remains visible without a
repeated question on every run. New evidence or an explicit request may reopen it.

There are two ways to obtain values, with a separate question of decision authority:

- **Compute from available data using existing rules.** The glossary generator
  takes names, groups and modes from its eleven source declarations and config
  rows, but reads translated values from `locale_src/<lang>/translation.json`
  (`scripts/translation-glossary.js:46–62,283–289`). Reproduction requires the same
  source and catalogue snapshot; this path makes no new translation decision.
- **Have an agent formulate values or propose a rule from evidence.** The
  context skill supports individual screen-based answers as well as reusable
  domain rules. A rule review can cover many records, whereas individual answers
  remain individual values. The skill's `next`/`submit` procedure does not require
  human approval of every row (`SKILL.md:64–94`); it relies on the agent's granted
  scope and input validation. This does not establish who authored every stored
  formulation historically.
- **Resolve decisions outside that authority with the owner.** This includes
  naming/canon choices even when candidate text can be read from data, as well as
  genuinely unclear meaning. Missing optional guidance alone does not require a
  question or block translation. An answer stays local unless evidence supports
  a reusable rule and its adoption is authorised.

Measured on the pinned t3 snapshot: `terms` contains 1032 entries and 29 928
language slots, `manual` 124 entries and 2792 slots. The total of 1156 entries is
before merging sections: six keys overlap, leaving 1150 distinct terms. The
generated modes are 939 `exact` and 93 `stem`. There are 105 rejected candidates
and no queued candidates. `translation_contexts.json` contains 857 distinct
formulations, 839 matching the 3559 source rows. `sources.js` supplies 23 domain
handlers (17 named directly and six generated for events) and one pattern rule.
These counts describe stored data, not how much human review each path required.

The old coverage figure 3494/3556 is dated 2026-09-04 (`WORKFLOW_NOTES.md:22`).
Subtracting today's 839 matching manual entries to get 2655 rule-covered rows
mixes snapshots; subtracting 3494 from 3556 gives 62 uncovered rows, not 62 human
decisions. Measure current coverage by resolution source and classify the reasons
for missing context before estimating operator work. The tracked
`locale_src/translations.csv` has an empty context column in all 3559 rows; actual
context application targets `.tradusco/booty/translations.csv`, a different file.

A rule is a repository record with a selector, a computable value function and
permission to return “no statement”. It must reproduce its result for the same
inputs. Rules make repeatable agent-authored logic reviewable; they are not an
escalation tier between computation and asking. A per-record answer does not
implicitly become a class rule. Review checks both the formulation and selector
behaviour, including distinct branches; one example cannot establish correctness
for every record, and dynamic formulations need not collapse to a small set.

Tradusco owns the following protocol for project-specific preparation rules.
The selectors and product meanings stay with the project; shared glossary
matching and format-validation semantics remain Tradusco responsibilities.

1. **Explain coverage.** Report attachment by class and resolution source,
   including unhandled records, conflicting candidates and values masked by a
   manual entry. Projects define classes; Tradusco provides the common report.
2. **Preview rule changes without writing.** Show old/new formulations and
   affected counts, examples from selector branches and access to affected rows.
   Include removals and unchanged effective values, not only new coverage.
3. **Apply separately against checked inputs.** Tie the preview to the rule and
   input revisions it describes; reject or recompute if either changed. Applying
   a change must not silently write results different from those reviewed.
4. **Separate changing rules from using them.** Rule editing is explicitly
   requested preparation work. Ordinary runs automatically apply existing rules;
   a separate write operation is not a new human-approval requirement on every
   run. Read-only inspection never writes, and agent answer submission uses its
   existing authority rather than pretending each answer is a reviewed rule.
5. **Keep current fill-only application in the first upgrade.** Existing rules
   fill empty nonmanual context values and preserve manual entries. Refreshing
   stored generated values after a rule change remains backlog.

Answers must be looked up before they are asked for. Prepare available glossary
data before context consumers that use it; other ordering follows declared data
dependencies, not a universal requirement to finish all naming decisions first.
Leila is recorded as `hero`/`stem` from `heroes/heroes.csv`, and the mistaken
item description is documented in the context skill's `references/map.md:92–95`.
But `isHeroName` itself reads the hero table directly (`scripts/lib/context.js`
inside that skill, `:104–123`): this failure proves a missing entity check, not
that rerunning the glossary generator would have fixed it. Both full-source
frequency mining and the selected changes can introduce glossary candidates;
both consult recorded decisions. Context inspection can feed new candidates
back into that same loop without blocking unrelated ready records.

Context has four levels: project-wide context, language context (both through
`load_context`, `lib/storage/filesystem.py:325`, combined at
`lib/TranslationProject.py:146-159`), the per-record `context` column, and
per-record per-language `context_{language}` columns, joined with `"; "` at
`lib/TranslationProject.py:677-684`. They accumulate rather than override: no
level replaces another today. A rule fills the record-level column, so a rule
and a hand-written per-language note reach the model together, and a rule that
contradicts a more specific level produces conflicting guidance instead of
losing. Whether any level should win is open; the accumulating behaviour is the
current one and must be stated before it is changed.

There are also two different resolution operations: the t3 provider chooses
manual → column → domain → pattern within the record-level context
(`scripts/lib/context.js:350–382` in the context skill), while Python accumulates
that result with project/language guidance. The current `context-apply.js:82–88`
keeps an already populated value for nonmanual results. The first upgrade
preserves that behaviour; provenance-based rule refresh and automatic downstream
regeneration remain backlog.

An accepted canon/context change can select affected unreviewed cells for
regeneration even when source text is unchanged. Scope may initially be explicit;
automatic dependency selection must explain its basis. Preserve editorial values
and the existing locale regeneration allow-list. Preparation that changes the
scope returns to P1 before dispatching requests. Optional inference work and
translation are distinct from read-only inspection and require appropriate run
authority; deterministic preparation does not invent terminology decisions.

Prefer explicit editorial recording at the moment of a supported correction.
`translation_fixes.json` already demonstrates that path. The first upgrade accepts
editorial changes through guarded edits or G2 back-sync; inference for changes
arriving outside them remains backlog. Existing progress, fixes and export state
remain the storage inputs. Progress is not
automatically a pure model-output log: back-sync writes imported values into it,
and recomputing export after progress changes does not prove what was last exported.

**Persistence scope for this upgrade.** Preserve the existing progress-first
resume behaviour and port t3's guarded edits. A failed export retries from saved
translations without another model call. General transactions, every possible
crash boundary and concurrent writers are deferred.

### Branches and edge cases at their point of occurrence

| At | Case | Required behaviour / explicit open boundary |
| --- | --- | --- |
| P0 | Existing catalogue and progress disagree, with no trustworthy baseline | Preserve both and report the mismatch; do not infer authorship. Resolve it through an explicit guarded edit or back-sync. |
| P1 | Source changes under a stable ID, or source-as-key changes | Translate the new source automatically after normal checks; preserve old source/key, translation and editorial history. Do not demand reapproval just because the source changed. |
| P1 | Record disappears or later returns | Preserve history. Retire it only from a complete source snapshot; on return, reuse the saved translation only when the exact source key matches. |
| P1/P5 | Host key was never managed by the selected extraction | Preserve it during export. |
| P1/P2 | Canon or context changes with unchanged source | Select affected eligible cells for regeneration under existing locale policy; retain editorial values. Explain selection and return through preparation before model execution. |
| P2 | Candidate was rejected or deferred previously | Reuse the decision for unchanged evidence. Do not repeatedly request the same naming decision; deferred work can be resumed explicitly. |
| P2 | No context, applicable term or reference exists | Continue with available guidance and report omissions. Do not manufacture product facts, require canon for every word or force reference-first translation. |
| P2 | Configured provider fails or guidance is structurally invalid | Report the failure distinctly from zero coverage; stop dependent work unless the provider was declared optional. Do not silently substitute partial generated output. |
| P2/P3 | Reference was machine-translated or its source changed | Machine origin is permitted and labelled honestly. Material for an outdated source is not presented as a current translation. Reference selection never changes which base text is translated. |
| P3 | One locale is missing from a multi-locale response | Preserve valid other locales; retry only unresolved eligible cells within budget. Outcome is partial if they remain unresolved. |
| P3 | Timeout, transient failure, refusal or request cannot fit | Keep distinct failure reasons and bounded continuation. Retain current input-aware/split policy pending evidence. **Open:** initial limits, nested retry accounting, handling one oversized phrase and refusal isolation. No invented fallback model or content rewriting to force completion. |
| P4 | Required placeholder is lost | The candidate is technically invalid; attempt bounded repair of automatic output, preserving successful cells. Exhaustion leaves unresolved work. |
| P4/P6 | Unusual script, length, source-equal text or glossary disagreement with editorial text | Report an applicable warning; no paid repair from a heuristic alone. Editorial text wins over glossary rules without a mandatory resolution step. Technical validation still applies. |
| P4 | Source or editorial text changes while the model is working | Old-source output cannot satisfy the new revision; editorial changes for the same revision win. Report superseded work and reconcile the next selection. |
| P4/P5 | CSV writing fails after progress was saved | On restart, repair the CSV from progress without another model call. Other crash boundaries remain backlog. |
| P5 | 97 of 100 cells are ready | Export the ready subset now; preserve unresolved host values and report partial completion. Stale values are not counted as fresh success. An adapter unable to export partially reports that limitation, without losing saved translations. |
| P5 | Compiler exits zero but drops a key | Report incomplete delivery from artefact comparison. Correct the delivery problem and retry P5, without another model call for saved valid translations. |
| P6 | Review proposal was prepared against a value that has since changed | Keep the original expected value; do not replace it with the live value when converting the proposal. Report the conflict and retain both editorial decisions when neither has established precedence. |
| P6 | A correction suggests a reusable rule | Record the candidate and evidence separately. Project-authorised decisions can accept it; an isolated correction does not automatically become a global rule. Cross-locale agreement is a clue, not proof against the source. |
| P7 | Interrupted request has no recoverable result | Resume from known persistence. A replacement request may be needed; do not promise zero duplicate provider charges. Never retranslate a known persisted result solely to retry export. |

### Worked round trip for review

This is an acceptance scenario, not a claim that it has been executed.

1. Connect an existing project; import known state and resolve first-import
   ambiguity according to the eventual P0 contract.
2. Add three strings and change one source string that previously had editorial
   text. P1 selects four new translations per requested locale and retains history.
3. P2 refreshes available deterministic guidance. One new string lacks context;
   it is reported, but translation proceeds. Available machine references are
   labelled as such.
4. One locale fails for one cell. Persist other valid results, retry only that
   cell within budget and export ready work. Report partial completion if needed.
5. A later resume completes the remaining cell. If the build fails, another
   delivery attempt uses persisted results and does not call a model.
6. An editor changes one value through guarded edit or G2 back-sync. Record its
   editorial origin when applying it; it survives regeneration and export for
   that source key even if the glossary disagrees.
7. Optional review records a correction and a context suggestion. Apply the
   authorised correction through guarded persistence; only an accepted context
   change enters future input preparation.
8. Another source change starts the next cycle automatically. The old editorial
   translation remains in history; it does not block the new translation.
9. Change canon without changing source: select affected unreviewed cells in
   allowed locales, prepare inputs and regenerate. Preserve editorial cells and
   host keys that were never managed by the selected extraction.

Run the round trip in the independent CSV/source-as-key fixture. Offline model
substitutes verify control flow and persistence; they do not establish translation
quality. Validate the provider boundary in the next real project after this slice.

The repository-level executable specification for this round trip is the
[small-shop acceptance scenario](ACCEPTANCE_SMALL_SHOP.md). It defines the
independent fixture, observable results and fault checkpoints
used to verify the selected vertical slice without depending on t3.

### Reconciliation with the existing records

| Process responsibility | Existing plan coverage | Required reconciliation |
| --- | --- | --- |
| Connection and sources (P0/P1) | G7, G11; F1–F4; R4–R6 | Preserve the current source-as-key workflow and separate catalogue-specific reading behind the project provider. |
| Selection and input preparation (P1/P2) | G1, G8; R1–R3; W1/W2/W5–W7; L6; glossary/context skills | Selection feeds the same operations in preview and execution. Machine guidance is permitted with provenance; optional input gaps do not become gates. |
| Model execution and observation (P3/P4) | G3; O2/O4; W3 | Expose failures, bound a run and preserve partial results without changing batching policy. |
| Validation and editorial preservation (P4/P6) | G4–G6; L1/L2/L5/L6/L9; F2/F4 | Port existing deterministic checks and glossary conformance; technical failures can retry, warnings cannot. |
| Persistence, export and recovery (P4/P5/P7) | G2/G5/G10/G11; O2/O3; R5; T1/T2; W3 | Preserve reviewed edits, partial export and current resume behaviour. |
| Reading and learning from review (P6) | G8/G9; context/glossary/review skills | Port ordered reading and decision queues; keep project meaning outside Tradusco. |

`WORKFLOW_NOTES.md` contributes input preparation, separate manual/generated
context, source authority, identity invalidation, transport checks and deriving
rules from editorial work. Its reviewed-only reference rule, blanket lint gate
and CSV-only correction advice are superseded for this rework by the agreed
process. Current-runtime descriptions remain historical facts, not implemented
promises of this design.

`REVIEW_SKILL_SPEC.md` remains evidence for the deferred review skill. It does not
add requirements to the first upgrade.

`GLOSSARY.md` supplies current merge/matching semantics; shared audit must agree
with prompt selection. `BATCH_BASELINES.md` supplies the current batching policy
and dated measurements, not approval to restore the cancelled output heuristic.
`INTEGRATION_GUIDE.md` describes existing CSV/PO use and explicitly notes that an
ID column does not yet change progress identity; P0/P1 must not promise otherwise
before the rework provides it.

### Existing decisions and mechanisms to retain

- Keep the current source-as-key identity for this upgrade. Stable-ID storage and
  migration remain outside its scope.
- `regenerateLangs` already gates regeneration in the t3 runner (historical
  `run.js:343` at the revision above). Preserve that protection. It is not a
  demonstrated prohibition of all automatic gap filling or new-source translation.
- Existing `--ensure-complete` and `sync_project_from_csv.py:202` provide gap
  recovery and placeholder quarantine. Extend them for failed replacements,
  per-locale outcomes, bounded accounting and interrupted writes.
- Explicit protected corrections already exist in `translation_fixes.json`.
  Their two-write durability gap remains; their existence must not be replaced
  by an authorship-inference framework.
- **Required rework, not an experiment:** remove unconditional reviewed-status
  claims for both `reference` and `examples` in `prompts/translation.txt:18–19`.
  Carry actual available provenance when presenting guidance. This follows the
  agreed policy, independent of R2 quality measurements. The prompt is unchanged
  during documentation review; its wording and associated checks belong with P2/P3.

### What prevents this workflow from becoming an implementation-ready task

Only three choices remain before implementation:

- the project-provider input/output shapes for glossary sources, context sources
  and host catalogue access;
- the package/entry-point location for the transferred Node operations;
- the exact G1–G11 delivery checklist against the small-shop acceptance test.

Identity redesign, migration, generalized persistence and experiments are not
prerequisites for this upgrade.

## Three layers

The proposed operation-by-operation responsibility map is in
[`REVIEW_ROUND2_TOOLING.md`](REVIEW_ROUND2_TOOLING.md#responsibility-map-concrete-t3-patterns).
It covers G1–G11 and the context, review and agent workflows, with provider inputs
and acceptance examples. It is pending agreement; the layer descriptions below
have not yet been rewritten to adopt its proposed boundary changes.

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

The 2/3 boundary has been sketched once already, in the wrong repository, and it
is not as clean as it looks from the file names. The t3 context skill is roughly
split along it:

| File | Contents | Layer |
| --- | --- | --- |
| `scripts/lib/context.js` | `.po` domains, config join, manual overrides, resolution order, queues | 2, with layer-3 leakage |
| `sources.js` | which column of which config means what | 3 |
| `context-ui.js`, `context-apply.js`, `context-report.js` | pipeline and report | 2 |

The leakage is real and has to be separated before anything moves. `context.js`
does locate the project root by `tradusco.config.json`, but it also hardcodes
`conf_src` and `server/conf` as the places configuration lives (`:41`), reads the
hero table by path and walks eight fixed ability slots per hero (`:104`) — that
is the product's data model, not a resolution mechanism. The shared `.po` helpers
are the same case: `scripts/lib/po.js:47` carries a list of catalog files that
count as narrative, which is a classification of this game's content sitting in a
library that otherwise only parses a format. The glossary side is cleaner —
`lib/glossary.js` reads `glossaryFile`, `locales` and `regenerateLangs` from the
config, and the project-specific part is the separate script that builds the
`terms` section out of the game's config tables.

So the move is not a file move. It is three separations done first: the game's
data access and its content classification into a project provider, gettext
reading into a format adapter (F4, T1), and only what is left — the resolution
order, the manual layer, the queues, the report — into the package. G1, G6 and
G11 all depend on that split having happened.

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

Tradusco is Python and the current t3 integration is Node. This establishes a
working implementation, not a Node requirement for every host. Existing format
readers and host toolchains should inform packaging; the standalone installation
contract and which operations require Node remain undecided.

Proposal: a `tools/` directory in the Tradusco repository with its own
`package.json`, published alongside the Python package. The contract between the
two languages already exists and is already used by both sides —
`tradusco.config.json` is read today by the Node integration. The Python engine
reads `<project>/config.json` (`lib/storage/filesystem.py:65`), so these are not
yet one shared configuration contract. Ownership, precedence and propagation of
settings between the two files remain to be specified. Python entry points and
Node packaging are proposals, not settled consequences of the current layout.

Duplicating common operations across projects has a maintenance cost: F3 records
an extractor defect fixed in the t3 copy but still present here. This supports
sharing the operation, without proving a particular language or packaging choice.

## What moves out of the integration

| Code | From | What it is |
| --- | --- | --- |
| G1 | `tradusco/run.js`, `preflight.js` | step orchestrator with skip flags, and the pre-run gate. Project binding is already in the config |
| G2 | `tradusco/sync-from-catalogs.js` | the return path Tradusco lacks: hand-reviewed catalog edits pushed back into `progress.json` and the CSV |
| G3 | `translation-status.js`, `translation-bakeoff.js` | move pre-run status; defer model bakeoff |
| G4 | `translation-scan.js`, `-markup-check.js`, `-script-check.js` | mechanical detectors, see L1 |
| G5 | `translation-apply.js`, `-patch.js`, `-unify.js` | guarded edits: expected `from`, idempotent, conflicts instead of silent overwrite |
| G6 | `translation-lint.js` | glossary conformance, see L2 |
| G7 | `translation-rekey.js`, `-sort.js` | move catalogue sorting; defer rekeying with identity migration |
| G8 | `translation-terms.js` | mining glossary candidates from English strings by frequency and mid-sentence capitalisation |
| G9 | `translation-catalog.js` | read one catalog across all languages at once, for review |
| G10 | `tradusco/validate_po.js`, `export_translation_json.js` | catalog validation, and `progress.json` → catalog export. The traversal and the merge rules are generic, the output shape is a project adapter |
| G11 | `scripts/lib/{csv,po,dotted-args}.js` | CSV and `.po` reading, the domain map, and dotted-placeholder normalisation (format/project-specific parts must be separated) |

Stays in the project: building the `terms` section from config tables, the
context source map, reading dialogue in script order, per-language style rules,
and review bookkeeping. Runtime coverage (`verify_runtime_coverage.js`,
`check-compiled-catalogs.js`) is a mixed case — see the transport section: the
gate is generic; compiler invocation, artefact reading and key normalisation
need adapters.

One cleanup on the way out: t3 currently has two implementations of context
filling — `scripts/translation-context.js` and the skill's `context-apply.js`.
The first is superseded and should be deleted rather than ported. It is not a
tidiness item: on 6 September the superseded script was the one reached for
first, and because it has no manual layer, the manual context was about to be
hardcoded into it. Two entry points for one step means the wrong one gets used.

## Run observability (O)

**Agreed completion/recovery behaviour (owner, 2026-09-07).** Retain successfully
persisted cells and retry only unresolved eligible work within the declared budget.
After budget exhaustion, report partial completion and resume only remaining work
on a subsequent invocation, reconciling source and editorial changes first.
Export the valid ready subset immediately without deleting unresolved/unrelated
host values; preserved old text is not a successful new translation. If export
or build fails after persistence, retry delivery without retranslating those
results. After interruption, continue from confirmed persisted work; a lost model
response may require another request. Completion of translation and delivery are
reported separately. This agrees behaviour, not implementation or numerical limits.

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
of unresolved cells after recovery, and — under regeneration — a record of which
requested replacements were validated and persisted, even if their text did not
change. Attempt failures alone do not establish the final outcome. Update the
wrapper alongside exit semantics so partial failure still permits ready-subset
export and bounded recovery; do not abort before those stages merely on nonzero.

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

**O4. A stalled call is never abandoned.**
`lib/llm/openai/OpenAIDriver.py:23` builds `ChatOpenAI(model=..., api_key=...,
base_url=...)` with no `timeout` and no `max_retries`. The effective value is not
a large default, it is no bound at all: LangChain passes `timeout=None` through to
the SDK client rather than letting the SDK's own default apply — constructing the
driver's client and reading it back gives `request_timeout None`, `max_retries
None`, and on the underlying client `timeout None`, `max_retries 2`
(`langchain_openai/chat_models/base.py:791`). So a request that never returns
never returns, and the SDK's two silent retries sit nested inside the retry loop
of `BaseDriver.py:99`, which has its own count and its own delay. Combined with
O1, a run that has stopped producing anything looks exactly like a run that is
working, indefinitely, and the operator's only recourse is to kill it and guess.

Needed: an explicit per-attempt `timeout`, a bound on total time and attempts for
one batch that accounts for the nested SDK retries rather than multiplying with
them, and a log line when either fires naming the batch and the elapsed time. The
same applies to any other driver that wraps a LangChain client.

What is not decided is the number. A single default for every driver and model is
not justified by anything measured — a flash model answering label batches and a
large model answering prose have different honest ceilings — so the value must be
overridable per driver and per model, and the default should be picked from the
observed distribution of successful request durations, which is O3's output. The
part that does not wait for that measurement is that the bound exists and is
announced; the failure this hides is not a slow model, it is a request that will
never return (C5).

## Cost (C) — deferred except glossary omission reporting

Keep current batching and cache behaviour in the first upgrade. Only C2's report
of eligible, included and omitted glossary entries belongs with the glossary move.

**C2. The batch glossary is truncated by entry count.** `lib/envelope.py`,
`_prompt_glossary`: entries are ranked by frequency and the loop breaks at
twenty, which `GLOSSARY.md` documents. Two things are wrong with it. The unit:
the constraint is tokens, and one entry with many inflections across many target
languages costs more than ten short ones. And the silence: nothing says when the
cap binds, so a term that should have been enforced simply is not, and the
translation that comes back looks like any other.

`BATCH_BASELINES.md` reports exactly 20 entries in a 50-row, 17-language request.
That establishes inclusion at the limit, not how many eligible entries were
dropped. Report eligible, included and omitted entries separately. Token-based
glossary budgeting remains a proposed change whose allocation and overflow
behaviour must be specified; visibility does not require first changing the cap.


## References, examples, regeneration (R)

**Agreed behaviour (owner, 2026-09-07; documentation only).**

- A changed source requires a new translation automatically, followed by the
  normal technical checks and export. Preserve the previous source/key,
  translation and editorial history; do not carry its protection onto the new
  source revision. Source change alone does not require human review.
- Editorial changes recorded by guarded edit or G2 back-sync take precedence over
  automatic translations and glossary rules for the same source key.
- Changes made outside those supported paths are reported as conflicts; inferring
  their origin is deferred.
- A glossary disagreement with editorial text is informational, not an approval
  gate or an instruction to overwrite. Technical validation still applies.

These decisions do not select a storage schema or authorise implementation.
Ambiguous identity mapping and competing editorial changes remain separate
conflict cases; they are not reasons to review every new source translation.

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
`lib/envelope.py:201`, `_examples`: neighbouring phrases with the same skeleton
are collected and their values are read from `dst_languages`, with no test of
whether those values were ever reviewed. Under `--regenerate` the CSV still holds
the previous machine translations, so those go into the prompt as models to
follow. This exposes regeneration to the text it is meant to replace; an effect
on the resulting translation has not been established. Building envelopes for
`uk` attached examples to 1206 of 3559 corpus rows, with 3564 example pairs in
total (measured 7 September).

What is not confirmed is the harm: nobody has shown that a run with these
examples produces worse output than one without. An offline envelope comparison confirms which examples would be sent; it
cannot establish whether those examples improve or harm output quality. Choosing
between reviewed-only examples (R6), disabling examples during regeneration, or
retaining them with an explicit trust policy requires a separate quality
comparison. New model calls for that comparison require an approved experiment
and spending budget. Independently, examples without review evidence must not
be described as reviewed in the prompt; that is a truthfulness issue, not a
quality hypothesis.

**R3. Regeneration is whole-language, all or nothing.** There is no way to
re-translate a handful of phrases — for instance the ones that just gained a
context line. A `--only-keys` selection, or selection by changed context, would
do it. Today this requires building a separate project.

**R4. The engine has no notion of a protected language.** Reviewed languages are
protected by the t3 orchestrator and by a `regenerateLangs` list in its config.
One misplaced flag overwrites a month of human review. This is a property of the
project and belongs in the Tradusco project config. Under the agreed behaviour,
protection prevents automatic replacement of editorial work for an unchanged
source; it does not prevent translating new or changed source text. A separate
policy forbidding all automatic work in a locale, if needed, remains undecided.

**R5. Dead keys accumulate and the engine does not prune them.** A phrase removed
from the source no longer appears in the phrase list, but its entry stays in
`progress.json` forever. Keeping it is the right default — a string can come
back, and its translation is worth money — but the engine has no counterpart: no
report of what is dead, no supported way to prune it, and no distinction between
"never translated" and "no longer needed" in any status output.

t3 has already built one, so this is a port rather than a design. `prune-progress.js`
separates retired, abandoned and stale entries and archives rather than deletes
them (`:13`, `:37`). Take the policy with it: what is preserved, on what evidence
a key is declared dead, and how an archived translation comes back when the
string does. Note that its notion of trust is dates of catalog review kept by
hand (`:130`) — see R6.

Recount before quoting the scale: after that pruning ran, each language's
`progress.json` holds 3629 keys against 3559 live phrases, 70 dead. The earlier
figure in this plan (5011 against 3556, 1509 dead) was also arithmetically
inconsistent: the difference is 1455. It is not a reproducible earlier baseline.

**R6. Preserve explicit editorial status.** Guarded edits and G2 back-sync record
editorial origin when they apply a value. The first upgrade does not infer review
or editorial status from an unexplained catalogue difference.

## Checks and lints (L) — transfer existing checks only

The first upgrade ports G4 and G6 plus checks required by their current behaviour.
L1, L2, L5, L6 and L9 are in scope; L3, L4, L7, L8 and L10 are deferred.

Two properties are checked during a run today: placeholders and lingui tags
(`lib/utils.py`, `placeholders_match` — see F4 for how narrowly that is defined),
and whether JSON scaffolding leaked into the output (`is_valid_translation`).
Everything else lives in the t3 integration, and all of it is generic apart from
the choice of markup delimiters.

**Agreed check policy (owner, 2026-09-07).** Technical failures such as malformed
output, missing required placeholders or invalid declared format can trigger
bounded repair of automatic results. Heuristics such as source-equal text,
unusual script, length or typography are warnings by default, not automatic paid
retry instructions. This narrows the earlier blanket gate principle in
`WORKFLOW_NOTES.md`. Project policy may promote a supported check explicitly;
editorial protection and informational glossary disagreements still apply.

**L1. Port G4's existing mechanical detectors.** Move the source-equal, script,
duplicate-translation, length, markup and whitespace checks already exercised in
t3. Keep format-specific delimiters in the project adapter. Unimplemented checks
listed in `WORKFLOW_NOTES.md` remain in `BACKLOG_TOOLING.md`.

**L2. Nothing verifies what the glossary is for.** Entries go into the prompt,
but no check confirms the term actually reached the translation. The t3
conformance check does this with a list of acceptable inflections, and it is the
only signal that an entry did anything. Without it a bad entry is invisible until
a human reads the catalog.

It cannot be ported as an equivalent of the engine's rules, because it is not
one. The check looks only at the `manual` section, skips `keep` entries, matches
only `s`/`es` suffixes in the source term (target forms use a separate heuristic),
and searches `near` in a ±40-character window
(`translation-lint.js:34`, `:92`, `:115`); the engine applies rules over both
sections and evaluates `near` against the whole phrase (`lib/envelope.py:40`,
`:57`). A lint that decides applicability differently from the prompt tells the
model it broke a rule it was never given — which matters most under L5, where the
finding is fed back as an instruction. So the shared piece is a single contract
for when a rule applies, with one set of fixtures exercised by both the Python
and the Node side; the conformance check is written against that contract, and L5
comes after it.


**L5. Check results need selective feedback.** Feed technical failures from
automatic output into bounded repair of affected cells. Heuristic warnings alone
do not trigger paid retries. The former claim of an order-of-magnitude saving
over human review was not measured and is not an acceptance criterion.

**L6. Nothing reports glossary entries that never fire.** An entry that matches
nothing is indistinguishable from an entry that works, so a glossary can be
inert in places without anyone noticing. Missing: a coverage report saying how
many phrases each entry matched over a run.

The share of `exact` entries is not that report, and the earlier reading of it
here was wrong. `exact` matches a phrase equal to the term (`lib/envelope.py:48`)
and therefore never fires inside a longer sentence — but a glossary of UI labels
is mostly made of terms that *are* whole phrases. Counted against the current t3
corpus: 939 of 1032 entries are `exact`, and once the sections are merged, 974 of
the 980 effective `exact` rules match at least one corpus row. Calling 91% of the
glossary inert was a misreading of the same number.

The report separates an entry that fires nowhere, one that fires only as a whole
phrase and one omitted by the twenty-entry prompt cap (C2). Target-language canon
gaps remain deferred under L10. Entry generators state a mode explicitly rather
than inherit `exact` by silence.


**L9. There are two override layers, and one of them duplicates the glossary.**
In t3 the export applies the per-locale fixes file first and the pinned-override
map from the config second, so the config layer wins over the fixes file
(`export_translation_json.js:221`, `:236`), and neither is compared against the
glossary. `Oblivion` was pinned to Latin in the config for every locale but
three; the glossary had since grown non-Latin canon for five more. Every export
quietly rewrote those five back to Latin, so a hand fix in the catalog survived
exactly until the next export, and the only reason it surfaced is that a
Latin-script lint was run afterwards.

The glossary guides automatic translation; editorial corrections determine the
text of specific cells for the source revision they address. Under the agreed
behaviour, editorial text takes precedence. Export must not silently undo it
through a glossary-derived or config override. A corrected whole-string label
may legitimately equal a glossary value; equal values are not competing authority.
Migration of existing competing override files is deferred. The transferred path
keeps guarded editorial fixes authoritative and reports adapter-level conflicts.

The check that remains is not "the same term has two different values". A
glossary entry can legitimately list several allowed forms, and its `scope` and
`except` narrow where it applies at all, so comparing two dictionary values would
report correct data as broken. Check the produced text against the rules that
actually apply to it — which is L2. For editorial text, a glossary disagreement
is informational: it neither blocks delivery nor triggers automatic repair.
Structural validation remains mandatory.

**L10 (deferred).** t3 has separate canon-gap and canon-import scripts, but they
are not in G1–G11. Their generic form remains in `BACKLOG_TOOLING.md`.

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

**F2. The glossary file is not schema-checked.** A typo in a mode does not
disable the entry, it widens it: `lib/envelope.py:42` handles `skip` and `exact`
and sends everything else to the regex branch, so `"mode": "typo"` behaves like
`stem` — `_Rule("Cat", {"mode": "typo"}, 0).spans("A Cat sleeps")` returns a
match. A rule meant to apply to one exact label then applies inside every
sentence that contains the word. In a 600 KB file this is not findable by eye,
and it fails in the direction nobody checks for.

**F3. `extract_translations_csv.py` loses phrases that changed.**
`_normalize_key` collapses whitespace (`extract_translations_csv.py:12`); the set
of existing keys is built from the normalised form (`:135`) and new phrases are
filtered against it (`:150`). `--collapse-whitespace` defaults to true (`:95`).
A phrase whose only change was line endings never enters the list and becomes
permanently untranslatable. The same defect existed in the t3 extractor and cost
two live strings that sat untranslated for six months.


**F4. The placeholder and tag syntax is hardcoded.** `lib/utils.py` defines
`_CURLY_TOKEN_RE` for `{token}` and `_LINGUI_TAG_RE` for lingui's numbered tags,
and `placeholders_match` — the only validation applied to every cell of every run
— is built on them. A project using ICU plurals, printf-style `%s`, `{{var}}`,
Rails-style `%{var}` or HTML gets either no protection or false rejections, with
no way to say so. This is the one place where Tradusco is already tied to one
project's conventions, and it sits in the hottest path in the codebase.

Configurable patterns are the cheap half and they are not the whole answer. A
format is more than a token regex: ICU plurals and selects have structure that a
regex cannot validate, and reading a catalog's keys, normalising them and
comparing two spellings of the same key (T2) are format questions too. So the
unit is a format adapter — extract and validate the structure, read keys,
normalise them — and a regex pair is the simplest adapter, kept as the default.

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
whole class. What is a project adapter is more than the command being wrapped:
reading the keys out of each stage's artefact is format work, and belongs in the
same adapter as F4.

**T2. Argument names are part of the contract, not a source detail.** When names
are generated upstream they reach both the runtime and the translator's screen.
They are rewritten at the build boundary (`{am.holy}` → `{am__holy}`), and the
runtime performs the same substitution again on lookup rather than reversing it
(`client/src/utils/l10n.js:9` against `scripts/lib/dotted-args.js:12`) — which is
the point: it is one normalisation applied on both sides, currently written
twice, and the two copies will drift. Make it one shared function. Any key-set
comparison has to apply it too, or it reports equal numbers of `missing` and
`extra` — the signature of comparing two spellings of the same key.

t3 has both as working prototypes (`check-compiled-catalogs.js`,
`lib/dotted-args.js`), which is where the generic version should be taken from.

## Orchestration (W)

**W1. Multi-language runs work in the engine and only halfway around it.**
`translate.py` accepts `-l fr,de,es` and sends phrases, context and glossary once
for all targets, which is the cheap path, and the t3 orchestrator already uses it:
`run-all.js:241` makes one shared multi-language first pass and then walks the
per-locale steps. What has not caught up is the rest of `run.js`, which still
assumes a single language and passes the raw string on — producing a path like
`locale_src/fr,de,es` — so export and apply are run separately. The generic
requirement is that the phase structure is the orchestrator's, not the operator's:
one shared translation pass, per-locale steps after it, and the language sequence
of W7 as part of the same plan. If the whole loop lives in Tradusco this cannot
drift.

**W2. Input inspection must be read-only.** Report the exact selected cells,
assembled inputs and missing guidance without model calls or persisted generated
artefacts. Distinguish measurements from estimates: future output length, cache
behaviour, retries and splits are unknown, so final cost and batch count cannot
be promised. The existing skip-based wrapper still writes helper artefacts and
does not satisfy this contract.

**W3. Filling gaps works, repairing damage does not.** `run.js --ensure-complete`
escalates the batch size downward (configured, then 20, then 10) and finally
moves to a fallback model on batches of 10, 5 and 1, and
`sync_project_from_csv.py` quarantines translations that break placeholders into
`progress._quarantine.json`. Both find *empty or invalid* cells. Neither finds a
cell that is populated but wrong — the abandoned-under-regeneration case from O2,
or the isolated language failure from O2a. With O3 in place, re-running exactly
the phrases a run failed on becomes a selection over the run log rather than a
search for holes that are not there.

For the first upgrade, current progress wins over older `failures.jsonl` entries:
a present valid value is successful, while an absent, invalid or quarantined value
remains unresolved. A separate final-outcome store is not required.

**W4 (deferred).** Long-run supervision moves with the deferred run skill. The
first upgrade adds only the O4 request timeout required to stop a hung call.

**W5. The pre-run gate should cover inputs, not just references.** Preflight
checks reference completeness and hashes the glossary. It does not report how
many rows are about to go to the model with neither context nor a glossary
entry. This is an input diagnostic, not a demonstrated predictor of rework or a
quality score. Seen in t3 on 4 September: 14 phrases from a new feature
went into a 17-language run bare, because the context step had not been re-run
after they appeared. The number was computable and nobody computed it.

**Agreed input preparation (owner, 2026-09-07).** Automatically run configured
deterministic context/glossary preparation for selected changes. Missing optional
guidance does not block translation and does not require an approval turn.
A configured provider failure is different: report it and stop its dependent
stage unless that provider was declared optional. Projects may declare required
inputs; the general tool must not inherit t3-specific requirements.

**W6. The incremental cycle is an acceptance criterion, not a separate feature.**
Everything is shaped for a full run, but the ordinary event in a live product is
different: a handful of new source strings appear, and they need the same four
steps — extraction, glossary, context, translation — over a much smaller set.
Measured in t3: three new UI strings cost roughly twenty operator actions, of
which three were the actual decisions — one glossary entry pair, three context
lines, one run. The rest was rediscovering the state of the pipeline.

The pieces are mostly present and unjoined, which is why this is not its own work
item. The engine already translates only what is missing
(`TranslationProject.py:648`), and the extractor already reports how many rows it
added (`extract-translations-csv.js:170`); what is absent is the list of *which*
keys those were, a way to tell the glossary and context passes "only these", and
one command that runs the four steps in order. Build it as part of G1, report it
through W5, and sequence the languages through W1 and R1. What this item
contributes is the test the orchestrator has to pass: three new strings, one
command, three decisions.

**W7. For new rows the reference languages are missing by construction.**
Preflight refuses to start when a reference locale lacks rows (`preflight.js:64`),
which is right for a stale reference and wrong for a new string: nothing has been
translated into anything yet, so every reference is incomplete on exactly those
rows. The escape hatch exists (`--no-reference-langs`), but the order it implies —
translate the reference locales first, then the rest against them — is written
down nowhere and is re-derived from a failed preflight each time.

**Agreed reference policy (owner, 2026-09-07).** Missing reference cells are
omitted and do not block ordinary translation. Translating reference locales
first is an optional project workflow, with its calls included in the run budget;
it is not a mandatory phase-order gate. Validate actual required inputs instead.

The harder half is trust, not completeness. `prompts/translation.txt:17` presents
references to the model as reviewed translations, and `WORKFLOW_NOTES.md` says a
reference must be a reviewed language — but a reference locale that was just
machine-translated in phase one is not reviewed, and feeding it to phase two
presents fresh model output to the model as human ground truth. Under the agreed
policy, current machine references are usable with honest provenance, without an
intermediate human-review requirement. Distinguish recorded editorial, machine
and unknown origin in the guidance supplied to the model and
operator. The original source remains authoritative. R2's separate quality
experiment may inform example policy; no blanket reviewed-only rule is imposed.

**W8. Refuse concurrent writes in the first upgrade.** G1 reuses the existing
lock-file mechanism for one project-run lock while a mutating workflow is active;
G2 and guarded edit commands refuse to start while it is held. A shared
cross-language write protocol remains backlog.

## Skills for agents

- **`tradusco-glossary`** — the candidate pipeline: present one candidate with
  all evidence, take a decision, run the conformance check before and after, and
  roll the entry back if it raised noise. Minus the term source, which is
  project-specific.
- **`tradusco-context`** — the same for context: one screen's worth of strings
  per turn, answer by file, validation and write. Minus the source map.

Only these two skills are part of the first upgrade. Run, QA, review and init
skills may be added after the transferred CLI workflow passes acceptance.


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
  Closed on 6 September for the ordinary path, and the answer was the first
  branch: the durable layer belongs inside the guarded edit.
  `translation-apply.js` now writes every applied edit to the catalog and to
  `translation_fixes.json` in one call. That is not yet durability, and G5 should
  not inherit the claim. The catalog is written first and the fixes file second
  (`translation-apply.js:89`), so an interruption between them leaves the edit in
  the perishable half only — and a re-run does not repair it, because the catalog
  already holds the new value and the edit is counted as `already` (`:70`).
  Repair after interruption has to be part of G5, not a consequence of ordering
  two writes. The same code shows a second gap for G5 to close: `from` is
  honoured only when supplied (`:74`), so the guarded mode is guarded only if the
  caller remembers to guard it. What the same day exposed is L9: the override
  layer and the glossary can disagree, and nothing notices.

## Order of work

1. **Glossary path:** move G8's candidate queue and decisions, G6/L2 conformance,
   L6 coverage and the shared glossary schema/matcher. Keep t3's term extraction
   script in t3. Make prompt selection and conformance use the same matcher.
2. **Context path:** move the manual/generated resolution order, rule preview and
   apply protocol, deferred/rejected queues and coverage report. Keep t3's source
   map and product-data readers in t3. Feed the resolved context into the existing
   translation envelope; preserve its current context-use instruction and prompt
   snapshots.
3. **Ordinary run:** move G1 and G3's status half with W2/W5 inspection, the
   incremental W6/W7 sequence, existing progress-first outcome/resume behaviour
   and the minimum O3/O4 logging and timeout needed to operate a run. Keep current
   batching and model policy; model bakeoff remains backlog.
4. **Review and delivery:** move G2, G5 and G9 for reviewed edits; G4's existing
   deterministic checks; G7's sorting half, G10 and G11 for reconciliation,
   validation and export. Preserve current editorial overrides and partial results;
   rekeying remains backlog.
5. **Acceptance:** complete the small-shop offline cases, run its live API check,
   then use the same provider contracts in the next real project. Record any
   capability that project actually needs as the next increment.

Deferred items above remain evidence or backlog and do not enter this sequence.

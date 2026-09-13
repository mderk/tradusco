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
meaning, identity conventions and host integration. No step assumes a game,
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
| P0. Connect | Read configuration and declared sources/formats; validate the integration. Reuse existing supported project state. Inspect legacy ambiguity only where an import could erase or misattribute existing work. | Supply source/catalogue access, locales, format support and providers. A plain phrase table is valid. Records may differ in domain and identity convention. Define identity uniqueness and ownership scope for each extraction; no separate corpus object is required. | Continue to P1 for configured and unambiguous data. Legacy migration is a separate bounded path, not a universal gate for every run or the whole design. |
| P1. Reconcile and select | Compare source and guidance revisions, explicit editorial records, host changes and available automatic-write evidence. Classify new, changed, unchanged, retired and unmanaged records. Include selected regeneration after canon/context/style changes, not only source changes. | Declare which extraction scopes participated and whether each result is complete or partial; supply identities, affected scope and regeneration policy. | Explicit work selection. Editorial values remain protected for the same source. No source/guidance change or explicit regeneration request means no model work; pending delivery may remain. |
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

**Identity and ownership contract.** A managed record has an owner namespace, a
local identity, the current source text and a revision derived from that source.
The stable Tradusco identity is `(owner, local identity)`; the source revision is
freshness evidence and never substitutes for identity. Where the host provides a
stable ID, use it as the local identity. A source-as-key integration uses the
exact identity emitted by its adapter and therefore creates a new identity when
that source key changes unless the project supplies an explicit unambiguous
migration. Domain and other descriptive metadata do not participate in identity.
A project may declare its domains as owner namespaces, and then the domain
participates in identity as the owner; a domain label carried on a record whose
owner is declared elsewhere does not. An integration with no ownership structure,
such as a plain phrase table, declares one owner for everything.

An owner namespace is also the extraction and completeness boundary. Every
extraction declares the owners it ran and, independently for each owner, whether
its result is complete or partial. Absence retires a previously managed record
only in a complete result for its owner. Absence from a partial result or from an
owner that did not participate changes nothing. Host records outside declared
owner namespaces are unmanaged and survive delivery.

P1 applies these rules without inference: an unseen identity is new; the same
identity and source revision is unchanged; the same identity with a different
source revision is changed and needs new translations; absence proved by a
complete owner snapshot retires it. A retired identity that returns is reactivated:
saved translations may be reused only for the same source revision, while a
different revision follows the changed-source path. History remains attached to
old revisions.

Emitting one identity more than once in a reconciliation needs two cases
separated. Repeated emission carrying the same source is a duplicate: collapse it
and report the count, since a record cannot be identified twice by its own
adapter. Repeated emission carrying different sources is a collision, because
nothing in the input says which source the identity now names. A collision never
merges records by source, domain or row order. It excludes the colliding
identities from the selection and denies that owner a complete result, so nothing
it manages can be retired on this run, while every other record of the same owner
proceeds. Stop the owner outright only when the collision makes its reconciliation
impossible, not when it is confined to named identities.

How a project draws its owner boundaries decides how many records it manages, so
it is a decision for P0 rather than for later migration. Measured on the pinned t3
snapshot, `locale_src/en/` holds 28 domains with 3864 msgid occurrences and 3521
distinct texts; 277 texts occur in more than one domain, heroine names such as
`Paula` and `Zoe` appearing at once in `heroes.po`, `items.po`, `quests.po` and
`offers.po`. One owner for the whole catalogue keeps today's source-as-key
behaviour, and those 277 texts stay collapsed into one record each with one
translation, which is exactly the limitation F5 describes. One owner per domain
separates them and adds 343 records that are then translated per owner. Neither
is a collision, and the contract does not choose between them; the project does,
before the first run.

Persisted translation state must therefore retain the logical identity, source
revision and value provenance needed by P1, P4 and P6. This is a logical record
contract, not a prescribed JSON or CSV layout. Existing phrase-keyed state lacks
those fields; its storage representation and legacy mapping are migration work.
Unknown or ambiguous legacy relationships are preserved and reported rather than
guessed. They do not block unrelated owners or new projects.

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
5. **Define replacement and downstream effects.** Replace values generated by
   the changed rule, preserve manual entries, and remove an obsolete generated
   value if resolution now yields no statement. Re-resolve other applicable
   rules before declaring a gap. Unknown legacy origin must not be guessed away.
   Report effective context changes separately from rule-file changes and use
   them to prepare the eligible regeneration selection described below.

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
keeps an already populated value for nonmanual results, so editing a rule does
not by itself refresh its previous output. The replacement protocol above is a
rework requirement, not a claim about current behaviour. The rework must add an
explicit operation that refreshes values produced by a changed rule. Each stored
context value therefore needs enough provenance to distinguish manual input from
generated output and to identify the producing rule and revision; without it the
operation cannot safely select values to replace or remove.

A rule change can alter context for many already translated records without
touching their source. A manual override or an unchanged computed result can also
make the effective input unchanged. Rule 1 does not apply, since the source is
unchanged, and P1 already selects regeneration after guidance changes under the
existing locale policy. What is missing is evidence: nothing records the context
under which a translation was produced, so a divergence between the current
formulation and the one in force at translation time is undetectable. Until a
guidance revision is recorded per persisted translation, regeneration scope
after a rule change can only be explicit, and this document must not claim
automatic dependency selection for it.

An accepted canon/context change can select affected unreviewed cells for
regeneration even when source text is unchanged. Scope may initially be explicit;
automatic dependency selection must explain its basis. Preserve editorial values
and the existing locale regeneration allow-list. Preparation that changes the
scope returns to P1 before dispatching requests. Optional inference work and
translation are distinct from read-only inspection and require appropriate run
authority; deterministic preparation does not invent terminology decisions.

Prefer explicit editorial recording at the moment of a supported correction.
`translation_fixes.json` already demonstrates that path. Infer presumed editorial
origin only for changes arriving outside it. Before adding baseline storage,
check whether existing progress, fixes and export state suffice. Progress is not
automatically a pure model-output log: back-sync writes imported values into it,
and recomputing export after progress changes does not prove what was last exported.

### Branches and edge cases at their point of occurrence

| At | Case | Required behaviour / explicit open boundary |
| --- | --- | --- |
| P0 | Existing catalogue and progress disagree, with no trustworthy baseline | Preserve both; do not invent authorship from timestamps. **Open:** first-import selection and reconciliation procedure. Subsequent editorial inference requires recorded automatic values. |
| P0/P1 | Identical source text appears under different identities | Keep separate `(owner, local identity)` records and histories. Domain labels alone are not identity. Legacy migration remains separate; reject an ambiguous mapping rather than silently merge. |
| P1 | Source changes under a stable ID, or source-as-key changes | Translate the new source automatically after normal checks; preserve old source/key, translation and editorial history. Do not demand reapproval just because the source changed. |
| P1 | Only identity/format spelling changes, with unchanged source meaning | Reuse only with an unambiguous declared migration. Whitespace removal is not proof. Without that mapping, a changed source-as-key identity is new. |
| P1 | Record disappears or later returns | Preserve history. Only a complete snapshot for the record's owner can retire it. On return, reactivate it and reuse saved translations only when the source revision matches; otherwise translate the changed source. Unknown legacy provenance remains a migration case. |
| P1/P5 | Host key was never managed by the selected extraction | Classify as unmanaged, not retired. Preserve it during export. A complete snapshot establishes retirement only within the identities that extraction previously managed. |
| P0/P1 | Two extractors emit the same local ID, or only one extractor participates | Equal local IDs in different owner namespaces remain separate. Repeating the same `(owner, local identity)` with the same source is a duplicate: collapse and report it. Repeating it with different sources is a collision: exclude those identities, deny that owner a complete result and continue with its other records. A complete result for A says nothing about absent B; do not retire B's records. |
| P1/P2 | Canon or context changes with unchanged source | Select affected eligible cells for regeneration under existing locale policy; retain editorial values. Explain selection and return through preparation before model execution. |
| P2 | A rule changes after preview, or now returns no statement | Recompute/review changed results before apply; replace or remove only attributable generated values and re-resolve fallback rules. Preserve manual context. Effective input changes, not a rule-file edit alone, determine the affected selection. |
| P2 | Candidate was rejected or deferred previously | Reuse the decision for unchanged evidence. Do not repeatedly request the same naming decision; deferred work can be resumed explicitly. |
| P1/P5 | Value differs from the known automatic value, unexplained by recorded automatic operations | Treat as presumed editorial, retain that distinction from confirmed review, and prefer it over automatic text for the same source. **Open:** accounting for automatic writers and competing editorial edits. |
| P2 | No context, applicable term or reference exists | Continue with available guidance and report omissions. Do not manufacture product facts, require canon for every word or force reference-first translation. |
| P2 | Configured provider fails or guidance is structurally invalid | Report the failure distinctly from zero coverage; stop dependent work unless the provider was declared optional. Do not silently substitute partial generated output. |
| P2/P3 | Reference was machine-translated or its source changed | Machine origin is permitted and labelled honestly. Material for an outdated source is not presented as a current translation. Reference selection never changes which base text is translated. |
| P3 | One locale is missing from a multi-locale response | Preserve valid other locales; retry only unresolved eligible cells within budget. Outcome is partial if they remain unresolved. |
| P3 | Timeout, transient failure, refusal or request cannot fit | Keep distinct failure reasons and bounded continuation. Retain current input-aware/split policy pending evidence. **Open:** initial limits, nested retry accounting, handling one oversized phrase and refusal isolation. No invented fallback model or content rewriting to force completion. |
| P4 | Required placeholder is lost | The candidate is technically invalid; attempt bounded repair of automatic output, preserving successful cells. Exhaustion leaves unresolved work. |
| P4/P6 | Unusual script, length, source-equal text or glossary disagreement with editorial text | Report an applicable warning; no paid repair from a heuristic alone. Editorial text wins over glossary rules without a mandatory resolution step. Technical validation still applies. |
| P4 | Source or editorial text changes while the model is working | Old-source output cannot satisfy the new revision; editorial changes for the same revision win. Report superseded work and reconcile the next selection. |
| P4/P5 | Crash occurs between writes, or another writer starts | Preserve completed work and recover a coherent state. **Open:** common writer coordination and recoverable persistence mechanism; per-file atomicity alone is insufficient. |
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
6. An editor changes one exported value. At the next reconciliation the
   unexplained departure from its known automatic value is presumed editorial;
   it survives regeneration and export for that source revision, even if the
   glossary disagrees.
7. Optional review records a correction and a context suggestion. Apply the
   authorised correction through guarded persistence; only an accepted context
   change enters future input preparation.
8. Another source change starts the next cycle automatically. The old editorial
   translation remains in history; it does not block the new translation.
9. Change canon without changing source: select affected unreviewed cells in
   allowed locales, prepare inputs and regenerate. Preserve editorial cells and
   host keys that were never managed by the selected extraction.

Repeat the same round trip in t3 and a small independent integration with
different catalogue/identity conventions, changing only configuration/providers.
Offline model substitutes can verify control flow and persistence; they cannot
establish translation quality. Paid quality experiments remain separate.

The repository-level executable specification for this round trip is the
[small-shop acceptance scenario](ACCEPTANCE_SMALL_SHOP.md). It defines the
independent fixtures, adapter variants, observable results and fault checkpoints
used to verify the selected vertical slice without depending on t3.

### Reconciliation with the existing records

| Process responsibility | Existing plan coverage | Required reconciliation |
| --- | --- | --- |
| Connection, sources and identity (P0/P1) | G7, G11; F1–F5; R4–R6; init | Identity/config/migration must support the selected process before dependent writes ship. Sampling syntax is not proof of format support. |
| Selection and input preparation (P1/P2) | G1, G8; R1–R3; W1/W2/W5–W7; L6/L10; glossary/context skills | Selection feeds the same operations in preview and execution. Machine guidance is permitted with provenance; optional input gaps do not become gates. |
| Model execution and observation (P3/P4) | G3; O1–O4 including O2a; C1–C5; W3/W4 | Define bounded outcomes; expose omissions, costs and failures. C3/C5 remain experiments, and R2 quality comparison does not block the agreed reference policy. |
| Validation and editorial preservation (P4/P6) | G4–G6; L1–L10; R4/R6; F2/F4 | Shared applicability; technical repair versus warnings. L7 remains an experiment. Glossary is not an authority to overwrite editorial text. |
| Persistence, export and recovery (P4/P5/P7) | G2/G5/G10/G11; O2/O3; R5; T1/T2; W3/W8 | One preservation protocol across writers; partial export and delivery verification; retirement separate from missing selected rows. |
| Reading and learning from review (P6) | G8/G9; L3/L4/L8/L10; context/glossary/review skills | Generic ordered reading and decision tracking, with project meaning outside Tradusco. No compulsory semantic review on the ordinary path. |

`WORKFLOW_NOTES.md` contributes input preparation, separate manual/generated
context, source authority, identity invalidation, transport checks and deriving
rules from editorial work. Its reviewed-only reference rule, blanket lint gate
and CSV-only correction advice are superseded for this rework by the agreed
process. Current-runtime descriptions remain historical facts, not implemented
promises of this design.

`REVIEW_SKILL_SPEC.md` contributes grouped multilingual reading, bounded scope,
defect categories, guarded corrections and resumable review. The rework must use
the shared write operation rather than mandate one source file format. Majority
agreement across locales is diagnostic evidence, not an automatic semantic
verdict; glossary disagreement cannot override protected editorial text.

`GLOSSARY.md` supplies current merge/matching semantics; shared audit must agree
with prompt selection. `BATCH_BASELINES.md` supplies the current batching policy
and dated measurements, not approval to restore the cancelled output heuristic.
`INTEGRATION_GUIDE.md` describes existing CSV/PO use and explicitly notes that an
ID column does not yet change progress identity; P0/P1 must not promise otherwise
before the rework provides it.

### Existing decisions and mechanisms to retain

- Identity direction is already recorded in `WORKFLOW_NOTES.md`: stable ID plus
  source-text hash where IDs exist, source-as-key where appropriate. That note
  groups the choice by corpus. This rework expresses the convention as the
  identity and ownership contract above, without introducing a corpus object.
  Persisted representation and legacy migration remain to be implemented.
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

The earlier U1–U6 list is now subordinate to this process:

- **P0/P1:** implement the identity/ownership contract and choose its persisted
  representation (U1/U3). Check existing provenance evidence before adding storage;
  legacy first-import reconciliation (U2) is a migration case, not a universal gate.
- **P3/P4/P7:** specify retry-limit semantics and recovery across write boundaries,
  including the explicit oversized-request/refusal branches (U4/U5).
- **Across the round trip:** choose the first supported integration capabilities,
  map every existing item to delivery or deferred experiment, then replace the
  old alphabetical work order with those dependencies (U6).

These are gaps to fill inside the process, not a new parallel checklist. The
task is ready when the selected round trip and its failure branches have one
unambiguous behaviour, responsibilities and offline acceptance evidence, with
remaining experiments explicitly deferred. Agreement on this document still
does not authorise implementation.

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
| G3 | `translation-status.js`, `translation-bakeoff.js` | pre-run state, and model comparison on a throwaway project |
| G4 | `translation-scan.js`, `-markup-check.js`, `-script-check.js` | mechanical detectors, see L1 |
| G5 | `translation-apply.js`, `-patch.js`, `-unify.js` | guarded edits: expected `from`, idempotent, conflicts instead of silent overwrite |
| G6 | `translation-lint.js` | glossary conformance, see L2 |
| G7 | `translation-rekey.js`, `-sort.js` | move translations to keys that changed only in whitespace, order catalogs by appearance |
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

## Cost (C)

**C1. Cache behaviour is invisible.** The prompt is already laid out for caching:
instructions and context first, phrases last (`prompts/translation.txt`).
Cache pricing depends on the selected model/provider and is not a fixed tenfold
saving. Nothing counts hits, so actual savings here are unmeasured. Printing
`input_cache_read` and `input_cache_write` from the response is enough.

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

**C3 (proposal). The glossary block breaks its own cache.** It sits after the
stable prefix and changes from batch to batch. Ordering phrases by which terms
they contain would make the block repeat more often across a run of batches —
though not identically, since the entry set still depends on which phrases
actually landed in each batch and on C2's cap. Neither the current loss nor the
saving is measured; C1 is what would measure both, and this should not be built
before it.

**C4. Whether large-batch-first still holds at 17 languages is worth
re-measuring — but not re-litigating.** Output-side batch estimation was specced
in `A6_BATCH_TASK.md`, implemented, and then deliberately removed in favour of
input-aware batching plus split-on-failure; `BATCH_BASELINES.md` records the
policy ("output size is not estimated or capped locally") and an arithmetic
sketch behind it: splitting one failed batch of 50 into five of 10 costs eight
requests against fifteen for always-small. The policy stands and this plan does
not reopen it, but that sketch should stop being cited as its proof. It models a
one-shot split into fives, and the engine does not do that: a failed batch is
halved recursively (`TranslationProject.py:546`, `:603`), and each attempt
carries the driver's own retries underneath it. A batch of 50 that only passes
once it is down to ten or fewer costs about seventeen calls per 150 rows against
fifteen for always-small, before counting a single inner retry. The break-even is
therefore not established in either direction; keep the current defaults because
nothing has shown them to be wrong, not because the arithmetic settles it.

What the 4 September run adds is a data point the baselines do not cover. They
were measured at up to 17 languages for size, but the failure statistics come
from single-language Spanish runs. At 17 targets, two batches of 82 failed hard
enough to exhaust three attempts before splitting, one of them on malformed JSON
at character 52913. That parse error alone does not establish truncation, and
the failure count does not establish which side of break-even this run occupies. The open question is whether the failure rate
rises with target count, which would move the break-even; the run log from O3 is
what would answer it, and until then the defaults should not move.

**C5 (proposal, not a finding). Batch size counts rows, and the cost of a batch
may be characters times languages.** `-b/--batch-size` is a count of phrases, so
the same number means a trivial request on short UI labels and a very large one
on prose. Nothing in the engine computes the quantity that would have to stay
inside the model's output budget: the sum of source lengths in the batch times
the number of target languages.

The observation behind it, from t3. A batch of 50 that happened to contain 20
hero backstories (median 832 characters, longest 1790) against 27 target
languages would ask for roughly 500 000 characters of generation in a single
structured response. That batch did not fail — it never returned. The same
phrases at `--batch-size 3` completed in 2623 seconds.

That is one run, and it does not establish the cause. Nothing in it separates an
exhausted output budget from an unbounded wait (O4), from provider-side
throttling, or from an ordinary slow response nobody was willing to sit through;
the specific batch was not recorded, so neither the elapsed time nor the
character figure is reproducible from anything on disk. The `parse_error` records
in `<lang>/failures.jsonl` were read as corroboration and are not: that category
also covers a response missing a language block, and `TranslationProject.py:267`
files model-level failures under it as well (`failure_reporting.py:24`). And this
is the heuristic that `A6_BATCH_TASK.md:3` recorded as implemented and then
deliberately removed, so bringing it back needs evidence, not a second opinion.

Where it sits against C4: C4's policy assumes a failed batch fails quickly and
cheaply, and a request that exhausts the output budget would not — it would hang
or burn the full generation before returning unparseable text, and the split
retries would pay it again. If that mechanism is confirmed, the two are
compatible: output-volume sizing as a guard against requests that cannot fit at
all, split-on-failure for everything merely large. Until it is confirmed, C5 is
not a decision.

What would make it one: the same set of short, long and mixed rows run across
several batch sizes and target-language counts, at a fixed model, provider and
retry policy, comparing cost, wall time, successful cells, actual output tokens,
`finish_reason` and the full retry chain. All of that is O3's log. O4 comes
first, because without a bound the "never returns" case cannot even be
distinguished from a slow one.

## References, examples, regeneration (R)

**Agreed behaviour (owner, 2026-09-07; documentation only).**

- A changed source requires a new translation automatically, followed by the
  normal technical checks and export. Preserve the previous source/key,
  translation and editorial history; do not carry its protection onto the new
  source revision. Source change alone does not require human review.
- For the same source revision, confirmed or presumed editorial changes take
  precedence over automatic translations and glossary rules. A change from the
  last known automatic value with no recorded automatic operation explaining it
  is presumed editorial. Preserve that distinction from confirmed review.
- Track automatic writes, including export and correction scripts, so that the
  inference has a comparison point. The present integration does not reliably
  establish authorship. Without a baseline, first-import provenance is unknown.
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

**R6. There is no review status below the level of a language.** R4 protects a
whole locale and R5 decides whether a key is alive, but neither says whether a
particular cell was ever read by a human. t3 carries that evidence by hand as
per-catalog review dates (`prune-progress.js:130`). Under the agreed behaviour,
record confirmed review separately from presumed editorial changes inferred
against known automatic writes. Protection applies to the corresponding source
revision. Keep source freshness and active/retired membership separate from that
provenance; the exact schema is not selected. Historical translations remain
useful even without review. Reference/example reuse policy still belongs to R2
and W7; presumed editorial authorship is not proof of human-reviewed quality.

## Checks and lints (L)

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
legal. Expose applicable checks in offline audit and in the run. Under the agreed
policy, only technical failures or explicitly promoted checks trigger bounded
repair; ordinary heuristic findings remain warnings. Reusing input context does
not imply that another model call is free.

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

**L3. Lints have no baseline.** The output is correctly shaped as "places to
read" rather than a verdict, but with no recorded baseline every run reprints the
same accepted places. Recording what was accepted leaves only new noise visible.

**L4. There are no per-language style rules.** Informal address in Russian,
politeness level in Japanese, Du versus Sie in German — every project reinvents
this. A declaration per locale in the project config, plus a hook for a project
script, would cover most of it.

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

What the report has to separate, because these have different cures: an entry
that fires nowhere at all; one that fires only as a whole phrase and so cannot
help inside a sentence; one that fires but has no form for the target language
(L10); and one that fired but lost the twenty-entry cap in the prompt (C2). Only
the first is a dead entry. Alongside it, entry generators should still state a
mode explicitly rather than inherit `exact` by silence.

**L7 (proposal). A name in the right script can still be in the wrong form.**
Substituting a
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

Why this is a proposal and not a finding. The 2.4% figure comes from a detector
that is itself narrow — it selects names by `group === "hero"`, tests inflection
by trimming the last letter, and reads only the fixes file
(`translation-inflection-check.js:36`). A number produced that way shows that
something was wrong in Ukrainian; it does not establish the detector's precision
or recall, and a grammatical check that is wrong in either direction is worse
than none, because L5 would feed its findings back to the model as instructions.
Before this is built, a labelled sample: a few hundred occurrences per language
judged by hand, against which the detector's error rate is stated. Then it is a
language plugin.

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
kind was sitting unnoticed in sixteen string-and-locale pairs, one label (`Find
Match`) accounting for ten of them. The class is not particular to games —
naming a control inside another string is standard UI writing, and vendor style
guides tell writers to do it.

Quoting is only the first heuristic for finding the reference, and the plan
should not promise more. The t3 detector reads quoted spans and separates UI from
prose by a catalog list (`translation-quote-check.js:44`), which misses every
reference written without quotation marks and depends on a project's own idea of
what prose is. The general form is that the project supplies the link between two
keys — from its own markup, from a reference syntax, or from a classifier — and
the engine checks that the target contains the linked key's translation. The
scope of a key comes from the domain column of F1.

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
How existing override files migrate into this rule remains to be designed.

The check that remains is not "the same term has two different values". A
glossary entry can legitimately list several allowed forms, and its `scope` and
`except` narrow where it applies at all, so comparing two dictionary values would
report correct data as broken. Check the produced text against the rules that
actually apply to it — which is L2. For editorial text, a glossary disagreement
is informational: it neither blocks delivery nor triggers automatic repair.
Structural validation remains mandatory.

**L10. An entry that fires can still have nothing to say in the target
language.** L6 counts whether a term matches phrases; it does not ask whether the
entry carries a form for the locale being translated. An entry with canon for
eight locales out of twenty-nine provides explicit target forms for eight, not
twenty-nine. The entry may still be sent with reference-language forms
(`lib/envelope.py:86`, `:139`); absence of a target form is not necessarily
absence of the whole entry. Prompt inclusion also does not prove conformance.
Two things are generic: a report of missing forms per entry per target language,
and an import path for confirmed canon so the answer, once found, is written back
to the glossary rather than to a catalog. t3 needed both and built them
separately (`translation-canon-gaps.js:47`, `translation-canon-mine.js:46`). What
does not generalise is how a candidate form is found — reading the reviewed
Russian column, or preferring non-Latin script, is a guess about this corpus and
stays in the project.

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

**F5. Phrase-as-key is the only identity, and it decides more than it looks.**
`progress.json` is keyed by the source string, which `GLOSSARY.md` notes makes
one thing impossible outright: two identical source phrases that must translate
differently in different places cannot be represented at all, no matter how good
the glossary or the context is. `WORKFLOW_NOTES.md` works the general problem
through and lands on stable id plus a hash of the source text stored beside the
translation — gettext's fuzzy flag, computed explicitly. The notes choose the
convention by source group because config-derived strings usually have IDs and
UI strings usually do not. The identity and ownership contract above makes that
choice explicit without requiring a corpus object: owner plus local identity
identifies the record, while the source-derived revision invalidates translations.
F5 implements that contract for new state and supplies explicit migration tools
for legacy phrase-keyed projects. It must not infer mappings from equal source,
domain labels, whitespace normalisation or row order. This remains prerequisite
to an honest implementation of R5 and rekeying.

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

That selection needs one thing the failure log does not currently give: the final
state of a key-and-language pair, as opposed to the history of attempts on it.
`failures.jsonl` is append-only, so a phrase that failed in March and was
translated in April is still a failure record, and the t3 workaround reads every
record it finds without checking for later success
(`prune-progress.js:160`). A log of attempts is the right primitive; the report
built on it has to resolve each pair to its outcome.

**W4. Long runs have no supported shape.** A full regeneration takes hours. There
is no heartbeat, no way to ask a running job where it is, and no guidance on
detaching. Each project rediscovers this.

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
intermediate human-review requirement. Distinguish confirmed review, presumed
editorial, machine and unknown origin in the guidance supplied to the model and
operator. The original source remains authoritative. R2's separate quality
experiment may inform example policy; no blanket reviewed-only rule is imposed.

**W8. Nothing coordinates two writers of the same files.** The engine writes
`progress.json` under a lock and through an atomic replace
(`lib/storage/filesystem.py:206`); the back-sync prototype that G2 would port
writes the same files directly (`sync-from-catalogs.js:85`). Today the collision
is avoided by the operator not doing both at once. Once the orchestrator can
start a long run and a person can push catalog edits back, that stops being a
convention. Either the write protocol becomes shared across both languages, or
concurrent operations are refused outright — but it has to be decided before G2
lands, not after the first corrupted progress file.

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
  syntax — samples suggest a format adapter but do not prove support for unseen
  structures or settle F4.
- *Asked:* which existing translations have editorial protection for their
  source revision (R4),
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

- **The agent answers with a file, not a pipe.** This is the t3 workflow's
  preferred handoff. Approval behaviour depends on the agent host and permissions;
  neither pipes nor answer files imply a universal approval rule.
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

**Provisional dependency sketch, not an approved execution sequence.** The
consolidated closure register in `REVIEW_ROUND2_TOOLING.md` lists what must be
decided before this order is rebuilt. The agreed behaviour blocks above take
precedence; estimated effort and future measurement do not authorise coding.

0. **O4**, ahead of everything else, and only O4. A run that hangs cannot be
   distinguished from a run that works, and every measurement below is taken on
   runs. Before implementation, define the unit of an attempt, the budget across
   retries/fallback/splits, timeout behaviour and configuration precedence. The
   size of that change has not been estimated from an agreed contract.
1. **O3 first, then O1, O2, O2a.** O3 leads because every other item on this list
   is a question about a request or a response, and none of them can be answered
   today. O1 and O2 need final persisted outcomes and compatible wrapper handling,
   not just surfacing failure records; effort has not been established.
   O2a — retrying a language whose block was lost — is
   the one place in this group where phrases are being abandoned for good.
1a. **C5's measurement**, once O3 and O4 are in. Not the implementation: the
   comparison that decides whether output-volume sizing is needed at all, since
   the same heuristic was already built and removed once.
2. **W5, W2** — the pre-run gate reporting context and glossary coverage, and a
   read-only inspection that separates measured inputs from cost estimates.
   W5 also has to produce the added-and-removed key list that the
   incremental cycle of W6 runs on.
3. **F5's representation for new state**, before the formats. The identity and
   ownership contract is settled above; its persisted representation changes what
   F1's column hangs off, what G2 syncs against and what R5 can honestly say about
   a dead key. Legacy migration stays late.
4. **F4** — the format adapter, starting with placeholder patterns out of the
   engine. It is the one coupling to a single project already inside the engine,
   so it should not survive the first item of decoupling work, and T1 needs the
   key-reading half of the same adapter.
5. **F1**, the domain column — unblocks more of the rest than anything else.
6. **L2's rule contract, then L6, L1**, with technical failures eligible for
   bounded repair and heuristic findings reported as warnings.
   The contract leads because a lint that decides applicability differently from
   the prompt is what makes L5's feedback wrong. Repair policy must distinguish
   warnings from failures; savings are unmeasured.
7. **T1, T2** — the transport gate. Cheap, mechanical, and it covers a failure
   that is invisible by construction: correct translations that never reach the
   product.
8. **G2 and G1 together, with W6, W7 and W8 inside them.** The return path and
   the orchestrator: the incremental cycle is the orchestrator's behaviour, the
   reference phase order is its sequencing, and the write protocol has to be
   settled before the return path ships. Without this pair any second project
   starts by losing its human review.
9. **F5's legacy migration**, using the representation introduced in step 3.
10. **Skills** for glossary and context, together with the layer-2 halves of both
   subsystems — which begins with separating the project's data access and
   content classification out of what is being moved. Then `tradusco-init`, which
   is what turns all of the above into a second project that costs a day instead
   of a month — and whose `check` half is worth having before that, since it is
   the only thing that tests whether the decoupling actually held.

Separately: **R2 input inspection is already confirmed; its quality effect is
not.** Offline envelope inspection can be repeated without O3 or model calls. A
controlled quality comparison is separate work and needs an approved experiment
if it requires paid calls. Do not treat a prompt diff as evidence of improved or
worse translations.

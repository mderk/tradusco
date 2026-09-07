# Second review: product direction and decision readiness

> **Status: discussion draft (2026-09-07). Not approved for implementation.**
> The user requested repeated document reviews and an independent assessment of
> whether the project is solving the right problem. Recommendations below are
> proposals, not additions to the accepted implementation scope.

Read alongside `REWORK_PLAN_TOOLING.md`. `REVIEW_RESPONSE_TOOLING.md` records the
paused O4 experiment and where its patch is preserved. No implementation should
resume until explicitly requested. No push, history rewrite or paid experiment
is authorised by this document.

## Independent assessment

The plan is a useful inventory of observed problems. It is not yet a coherent
product release plan. It mixes three different investments: translation quality,
preservation of editorial work, and reusable integration infrastructure. Counting
how many t3 scripts can move does not tell us which investment will reduce the
operator's work or improve a release.

The owner has confirmed the goal: a standalone tool usable in other projects,
enriched with reusable patterns and workflows learned in t3, including
deterministic operations. Project-dependent tools are necessary and remain a
supported layer. My earlier suggestion that this might primarily be an internal
t3 utility is superseded. Generalising the process is an intended outcome, not
an optional investment to justify from scratch.

My recommendation is to organise that generalisation around dependable
translation change workflows: explain what changed, translate the intended
cells, preserve reviewed work, recover incomplete work and prove what reached
the host catalogue. Each workflow should be usable through a documented contract
without editing Tradusco. Existing t3 scripts supply evidence and prototypes;
their current file boundaries do not by themselves define that contract.

The existing findings support this direction: regeneration can leave old values
looking complete (O2), edits can be lost between two writes (G5), back-sync can
race the engine (W8), and failed compilation can lose translated keys (T1). These
are problems even if the model produces excellent translations at negligible cost.

Conversely, a long list of heuristic lints does not demonstrate translation
quality. A clean mechanical report can accompany an incorrect meaning or voice.
Saving input tokens is also not necessarily the main saving if reviewing and
repairing the result takes much more operator time. The latter has not been
measured here and must not be presented as a proven cost breakdown.

## Corrections made in this pass

Only factual and internal-consistency corrections were applied to the plan:

- R2: 1206 rows out of 3559, containing 3564 example pairs; an offline prompt
  comparison cannot establish output quality. Corrected the handoff too.
- C4: removed the surviving claim that the historical failure rate is inside a
  break-even point that the preceding paragraph explicitly says is unknown.
- R5: the older count was arithmetically inconsistent, not a valid old baseline.
- L9: fixes are applied before config overrides, not the other way around.
- L2: `s`/`es` describes source-term matching, not target-language morphology.
- L10: missing a target form does not imply the whole glossary entry is omitted;
  reference-language forms can keep it in the prompt (`lib/envelope.py:86`, `:139`).
- Packaging: the engine reads `<project>/config.json`, not the Node integration's
  `tradusco.config.json` (`lib/storage/filesystem.py:65`). The shared configuration
  contract still needs to be defined.
- G11 and the transport summary: dotted-argument normalisation is not CLI argument
  parsing; transport adapters include artefact reading and key normalisation.
- O4: removed the unsupported "few lines" estimate. The existing experiment is
  evidence of unresolved choices, not evidence for its own design.

## Original decision findings (historical; status below)

This table records the initial review, not ten currently open decisions. The
consolidated closure register below is the current remainder. In particular,
D4's reviewed-only default was superseded by the owner's reference policy.

Paths prefixed `C/` refer to `/Users/max/Documents/projects/t3/client`; other
paths refer to this repository. Plan section IDs are used because line numbers
move during document reviews.

| ID | Issue and evidence | My proposal, pending agreement |
| --- | --- | --- |
| D1 | O4 says "one batch" without defining whether its budget includes SDK retries, format fallback, model fallback and split children. `BaseDriver.py:178` and `:193` can make two calls within one loop iteration; Gemini's installed LangChain version hardcodes two attempts in `chat_models.py:143`. | Define request attempt, logical batch and run separately. State which limit is inherited by split children and what exhaustion does before selecting values or a config schema. Distinguish client cancellation from proof that provider billing stopped. |
| D2 | O2's nonzero exit changes the host workflow: `C/scripts/tradusco/run.js:94` throws, and translation at `:529` precedes ensure-complete's export/retry decision. | Specify partial success, fatal failure and cancellation outcomes with their host behaviour. Ship compatibility with the existing wrapper alongside the changed exit behaviour; do not wait for the full G1 port. |
| D3 | R6 proposes one `reviewed/stale/retired` status, but a reviewed translation can belong to a retired source; freshness and trust are independent. R2 and W7 require trust while R5 requires source membership. | Describe review evidence, freshness and active/retired membership independently, without deciding storage schema yet. Do not discard a useful machine translation solely because it was not reviewed; make reuse policy explicit. |
| D4 | W7 would gate whether a phase order was followed even though R1 permits missing references. It only tells the operator that references were unreviewed, while `prompts/translation.txt:18` still tells the model they were reviewed. | Gate properties of the selected inputs, not how they were produced. Omit unreviewed references by default; if explicitly allowed, describe their trust honestly to both operator and model. This proposal still needs agreement. |
| D5 | R6 mandates reviewed examples before R2's quality comparison has decided the policy. L9 prohibits canon in any override, although a reviewed whole-string fix may legitimately equal the canonical label. | Separate truthfulness and preservation requirements from reuse policies. A checked label may exist as both a whole-string correction and a glossary term; prevent contradictory authorities rather than prohibit every duplicate value. |
| D6 | L1/L5 turn heuristic findings into paid retry instructions, while L3 describes findings as places to read. Source-equal translations, typography and length can be legitimate. `C/scripts/translation-lint.js:64` uses approximate stems. | Separate structural failures from review warnings. Only agreed high-confidence checks may trigger bounded repair; record accepted exceptions against the text/rule version, and do not infer approval from a clean lint. |
| D7 | O3 records outcomes but does not say whether "success" means parsed, validated or durably saved. `TranslationProject.py:483` saves locale progress files and only then the CSV at `:490`. A crash can separate these events. | Distinguish attempted, validated and persisted results. A retry list must reconcile with current storage and input revision; an old success event must not legitimise a newer source or glossary revision. |
| D8 | W2 promises cost and batch counts before responses exist; recursive splits, cache hits and output lengths are unknown. W5 treats missing context/glossary as a predictor of rework without measuring that relationship. | Report deterministic selection and assembled input separately from estimates, assumptions and unknowns. Coverage is an input diagnostic, not a quality score. Plan/dry-run must be read-only, including helper artefacts. |
| D9 | Packaging assumes every host's toolchain is Node and that the alternative never pays. The repository already supports CSV-first projects (`INTEGRATION_GUIDE.md:80`) and has Python PO readers. This does not prove Python is preferable either. | Standalone installation and extension contracts are part of the confirmed goal. Decide which operations require Node and how Python-only use remains possible before selecting packaging. Keep host adapters callable through a documented boundary; validate it on different catalogue conventions. |
| D10 | The order says G1/G6/G11 depend on removing project coupling, but puts that separation in the final skills step. R6, R3, G5 and L10 have no explicit place despite dependencies elsewhere. `init` still claims sampling placeholder patterns settles F4 after F4 was broadened to structural adapters. | Place prerequisite contracts before their consumers; assign items to a release, experiment or explicit deferred list. `init` should propose a detected adapter and validate it; sampling cannot prove support for unseen syntax. |

## Consolidated closure register

**Original-author review update:** see the response below before using U1–U6.
Identity direction, locale regeneration protection and repair prototypes already
exist; the register must not reopen them. Legacy import is a migration branch.

**Reading-order update:** the owner requested a complete workflow first, then
reconciliation with notes. Start with the end-to-end workflow in
`REWORK_PLAN_TOOLING.md`. U1–U6 below are its unresolved implementation-contract
questions, not a replacement for understanding and reviewing the full process.

Whole-document consistency pass, 2026-09-07. This is the finite remainder for
the present plan, not a new feature inventory. Product behaviour already agreed
is not reopened here. A code reference from earlier reviews is historical
evidence; this pass did not rerun the corpus measurements or paid experiments.

### Confirmed and no longer awaiting a policy decision

- Standalone Tradusco with a necessary project-dependent layer.
- Changed source gets a new automatic translation; prior editorial work stays
  with its old source revision and history.
- Confirmed/presumed editorial changes win over automatic text and glossary
  rules. Glossary disagreements with editorial text are informational.
- Preserve successful cells, bound retries, export ready subsets and recover
  delivery without retranslating persisted results.
- Prepare deterministic inputs automatically; missing optional guidance is not
  a gate, while configured provider failures must be explicit.
- Machine references are permitted with honest provenance; reference-first
  ordering is optional. Technical failures and heuristic warnings differ.

### Six remaining decisions

| ID | Decision and why it is still open | Recommendation to make concrete next | Closure evidence |
| --- | --- | --- | --- |
| U1 | Standalone boundary, installation and config ownership. The Three layers/G inventory still leaves reading and review bookkeeping wholly in t3; Language and packaging offers two runtimes but no settings precedence. | Adopt the responsibility map as the operation boundary; keep existing implementations where suitable, define one resolved configuration with explicit ownership and precedence. Do not require a plugin framework or rewrite merely for uniformity. | Installation/use example outside t3, provider inputs/outputs, and a conflicting-setting example with one unambiguous result. |
| U2 | Initial import and automatic-write provenance. R6's agreed editorial inference requires a baseline that legacy data lacks. Export, fixes and back-sync can otherwise manufacture presumed edits. | Specify how the first baseline is established without inventing authorship; thereafter account for every supported automatic writer. Preserve unknown legacy origin. Agree how divergent first-import values and two competing editorial changes are resolved. | First import with matching and divergent values; automatic export followed by an external edit; competing editorial changes. |
| U3 | Identity and migration boundary. F5/G7 cannot distinguish same-text occurrences or safely infer identity from whitespace; source revision is now required by agreed behaviour. | Define identity per corpus and source revision separately; preserve legacy identifiers on import and reject ambiguous mappings. Decide which migration is required for the first delivery instead of promising all migration later. | Two occurrences with identical source but different translations; a changed source under a stable ID; an ambiguous rekey with neither value lost. |
| U4 | Persistence and writer coordination. W8 and G5 require recovery across multiple writes, beyond a per-file atomic replace. | Start with one writer per Tradusco project and a specified recoverable write procedure; keep host-edit checks. Select the smallest mechanism meeting interruption cases, without choosing a database by default. | Interrupt each write boundary; restart restores a coherent state, preserves edits and does not report undelivered work as delivered. |
| U5 | Resource-limit semantics and initial values. O4 does not define the budget unit across SDK attempts, format/model fallback and split children. C2's proposed token allocation is also unspecified. | Define attempt/logical-work/run scopes and provisional overridable ceilings; validate accounting offline, then calibrate with separately approved measurements. Keep current glossary cap initially with explicit omission reporting unless its replacement is specified. | Nested retry/split example stays inside its declared limits; exhaustion yields partial work. Record why initial limits were chosen, without presenting the paused patch's numbers as approved. |
| U6 | First delivery scope, order and independent acceptance. The final plan order separates prerequisites from consumers and leaves several items unscheduled. | Select one round trip covering input changes, editorial preservation and recovery; put U1–U5 contracts before dependent implementation. Assign every plan item to first delivery, later delivery or experiment. Use t3 plus a small non-t3 integration with different identity/format conventions. | Complete item-to-delivery mapping and an offline end-to-end acceptance scenario for both integrations; separately identify any paid quality validation. |

U1–U6 are design work to close, not six questions that need immediate user
answers. Prepare concrete recommendations in dependency order, then seek only
the behaviour/scope decisions that require the owner. Exact function names,
JSON field spelling and routine implementation choices need not become approval
gates. No implementation is authorised by this register.

### Editorial corrections and reconciliation

Corrected in this pass without introducing new product policy:

- Removed the universal Node-toolchain claim, fixed cache-saving claim and the
  claim that exactly 20 included glossary entries proves omissions (C1/C2).
- Marked the old order provisional, removed its unsupported half-day estimate,
  exact-cost promise and blanket lint-feedback wording.
- Removed init's claim that sampled placeholders settle structural format
  support, and its blanket never-regenerate wording for reviewed locales.
- Qualified the t3 file-versus-pipe approval observation as host-dependent.
- Marked the original D table historical so its superseded reference policy
  cannot be read as the current recommendation.

Remaining mechanical reconciliation, after U1–U6 decisions: fold the adopted
responsibility map into the plan instead of leaving two competing layer
descriptions; replace the provisional order; update current guidance in
WORKFLOW_NOTES/REVIEW_SKILL_SPEC where it conflicts with the agreed policy.
Keep historical measurements and the original response labelled as history.
Translation and history cleanup remain deferred under the owner's instructions.

### Explicitly deferred; not blockers for document agreement

C3 cache grouping, C5 output-volume slicing, L7 grammatical-detector precision
and R2 example-quality effects require evidence. Retain them as experiments;
do not change the batching policy or claim quality gains to close this review.
Advanced language checks, additional formats and broad legacy migrations enter
the delivery mapping under U6; only migration required by its selected contract
can block that delivery. Existing t3 translation findings are a separate data
task, not a prerequisite to approving reusable-tool documentation.

### Coverage and limits of this pass

Read the current plan, second review and response, including the final work
order, init/skills and loose ends. Checked policy consistency across O/C/R/L/F/T/W
and G1–G11, and separated accepted behaviour from design proposals. No new code
audit, benchmark, model call, runtime test or independent integration was
performed. Earlier file references and dated counts were not recertified as
fresh measurements. `git diff --check` is a documentation formatting check only.

## Response to the original author's workflow review

The review correctly identifies an imbalance: the first P0–P7 draft described
state reconciliation in detail but underrepresented preparation decisions.
The main workflow is updated; this is not implementation authorisation.

| Finding | Disposition and change |
| --- | --- |
| N1/N2 | Accepted. P2 now includes evidence → decision → resolved input before translation; accept/defer/reject and “not a term” survive across cycles. No blanket approval requirement is added for missing optional inputs. Queue representation remains open. |
| N3 | Accepted. Unmanaged host keys are distinct from retired formerly managed records; preserve both appropriately, with corpus-scoped ownership. The historic 896 count was not remeasured. |
| N4 | Accepted. Guidance changes and explicit regeneration are P1 triggers; P2 returns to selection if new decisions change its scope. Editorial values and locale policy remain protected. |
| N5 | Accepted in direction, with a limit. Record supported corrections explicitly using the fixes prototype; infer only outside-tool edits. First evaluate available evidence. Progress is not necessarily pure automatic history because back-sync modifies it; current export computation cannot always reconstruct the last export. No new baseline store is mandated. |
| N6 | Partly accepted. Preserve existing regeneration allow-list. Historical runner lines 343–350 gate `--regenerate`; that does not establish a ban on all automatic translation in those locales. The agreed new-source behaviour is retained. |
| N7 | Accepted. Restore the recorded per-corpus identity decision: stable ID plus source hash where applicable, source-as-key otherwise. Multiple corpora per project are explicit; representation/migration remain open. |
| N8 | Accepted as missing credit/connection. P4 names existing ensure-complete and quarantine. They do not yet solve nonempty failed replacements, final per-locale outcomes or multi-file recovery; those gaps remain. |
| N9 | Accepted once; repeated occurrences in the received review are duplicates. Prompt truthfulness is explicit P2/P3 rework, not an R2 experiment. “Always false” overstates the evidence: some examples may actually be reviewed, but the unconditional claim is unsupported. No prompt code was changed. |
| N10 | Partly accepted. Cross-locale diagnostics belong in P4 as well as grouped reading in P6. Semantic disagreement detection may need inference; it is not established as a free deterministic check or the cheapest one. Majority agreement is evidence, not authority over source or editorial text. |
| N11 | Accepted. P0 does not force already configured or unambiguous projects through legacy migration. Conflicts block affected writes, not understanding or agreement on the whole process. |
| N12 | Confirmed checkout mismatch, corrected the proposed remedy. Current master is 14ad97c538; local common/main at 5778c5d also lacks scripts/tradusco. The main plan now provides Git-show commands against actual historical runner/correction commits without switching the dirty checkout. Other citations need per-file revision resolution. |

Remaining process gaps are narrower than the old register suggests: recoverable
writes; attempt-budget accounting; per-corpus identity representation and delivery
ownership; persisted decision queues; evidence needed for outside-tool edits and
legacy migration. Installation/config and delivery mapping remain specification
work inside that process, not reasons to reopen agreed behaviour.

## Proposed first useful delivery

The standalone direction is confirmed; the delivery scope and sequence below
still need agreement. Prefer one complete reusable user scenario over finishing
each alphabetical group:

1. New source strings: show the selection and input issues, translate only the
   intended cells, then report the persisted and exported result.
2. Interrupted or partially failed run: show what remains unresolved and resume
   that subset while preserving successful and reviewed cells.
3. Human correction: preserve it across export, sync, regeneration and restart,
   or report an explicit conflict instead of silently selecting a value.

O4 is a sensible prerequisite to live experiments. Minimal observability, outcome
semantics, selective repair, guarded writes and a host coverage adapter make
these scenarios possible. The delivery must be independently installable and
usable without t3 paths or game knowledge. Full identity migration, every format
adapter, all language lints and output-size optimisation need not block it.
That is a proposed scope boundary, not a decision to delete those items.

For each scenario, record operator actions and elapsed operator time, unintended
overwrites, unresolved cells presented as complete, and model cost per accepted
result. Retain a small fixed review sample for meaning and style. Set acceptance
thresholds before comparing results; do not invent numeric targets after seeing
them. A small independent integration with different catalogue conventions is an
acceptance test for the reusable contract, not a reason to pre-build adapters for
every format. It can initially be a test project rather than a production migration.

## Proposed reusable boundary

This section is my recommendation under the confirmed product goal, not an
approved API or command naming scheme.

- **Tradusco owns operations and their guarantees:** selection/diff, glossary
  resolution, context precedence, inspection, validation, guarded application,
  run outcomes, recovery, export orchestration and reports. Agent skills compose
  these same operations rather than maintain a second implementation.
- **Project providers supply meaning and host integration:** source extraction,
  origin/domain classification, context evidence, authoritative terminology,
  review decisions, compiler invocation and host artefact access. General
  catalogue/format adapters may be shipped by Tradusco; a project selects and
  configures one or supplies its own. They do not all need to be rewritten per host.
- **Boundary data must be explicit:** operation inputs, selected keys/locales,
  diagnostics, proposed changes, conflicts and persisted outcomes. A provider
  should not have to modify internal progress files just to participate in a
  normal workflow. The exact schema remains a design decision.
- **Deterministic operations must stand alone:** an agent can inspect, plan,
  check or apply an approved edit without invoking a translation model. Review
  and naming decisions remain distinct from mechanical execution.
- **Acceptance exercises the boundary:** the same workflow runs in t3 and a
  small non-t3 test integration, with only provider/config changes. Include a
  correction and interrupted-run recovery, not just successful extraction.

The key change I recommend to the plan's organisation is to give each G-item an
operation, its guarantee, its provider inputs and its acceptance example. That
turns "move this script" into an executable contract while keeping project
knowledge where the owner intends it to stay.

## Responsibility map: concrete t3 patterns

Discussion proposal following the owner's request to map the reusable patterns.
This is a behavioural boundary, not a package layout or a plugin framework.
“Adapter” can initially mean a configured existing reader or command with a
documented input/output. It does not imply a new interface hierarchy.

In the table, **workflow** and **engine** are both parts of standalone Tradusco.
Format and language support can ship with Tradusco without becoming universal
engine rules. A project selects that support and supplies its own product facts.
Acceptance examples below are proposed checks, not tests already executed.

| Plan items / observed pattern | Tradusco operation and guarantee | Project or selected adapter supplies | Boundary acceptance example |
| --- | --- | --- | --- |
| G1, W1–W7: incremental orchestration | Workflow selects explicit cells, prepares inputs, runs shared/per-locale steps and reports unresolved work. Inspection is deterministic; translation is a separately visible model step. | Source snapshot, extraction/build commands, locale/reference policy and integration paths. | Add three strings in a JSON-based non-t3 project; selection names exactly those strings and inspection makes no model call or writes. |
| G2, W8: back-sync | Workflow imports proposed catalogue changes through the same preservation/conflict rules as other writes; reports imported, unchanged, outside-corpus and conflicting cells. | Catalogue reader, origin of edits and available review evidence. A catalogue's location alone does not establish that its contents were reviewed. | An external correction can be imported without the adapter writing progress files; a concurrent change becomes a conflict. |
| G3, O3, W3: status and comparison | Workflow reports current outcomes and remaining work. A bakeoff reuses selection/reporting on isolated copies; model execution remains explicitly paid. | Comparison corpus, candidate model settings and quality judgements. | The same status operation works without a host compiler; an old failure followed by a persisted success is not offered for retry. |
| G4, L1, L3–L5: checks and repair | Shared diagnostics, accepted exceptions and bounded repair policy; engine response validation and offline audit use the same applicable checks. A warning does not itself authorise repair. | Format rules, locale support, product style and explicit exceptions. | An unchanged brand name can be accepted without disabling missing-placeholder detection for that locale. |
| G5: guarded corrections | Workflow previews and applies explicit changes with expected values, conflict reporting, repeatability and recoverable persistence. | Approved replacements and any host export adapter. Approval is supplied, not inferred from the presence of a proposal. | Change a value after review: the old correction conflicts instead of overwriting it. Repeat a completed correction: no new change. |
| G6, L2, L6, L10: glossary use | Shared applicability/resolution semantics, coverage and missing-form diagnostics; engine and audit agree which rule applies. | Canonical terms/forms, their provenance, opaque domains and selected language matching support. | The same entry applies to the same source/domain in preview, prompt assembly and audit, regardless of catalogue format. |
| G7, F3, F5: rekey and order | Workflow proposes identity migrations, detects ambiguity and preserves conflicting values. Catalogue presentation order is separate from identity. | Explicit old/new identity mapping, or a selected matching strategy; source occurrence order. | Two keys that become equal after whitespace removal are not silently merged. Reordering output does not change identity or review evidence. |
| G8: term candidates | Workflow gathers occurrences/evidence, presents candidates and records accept/defer/reject decisions. | Candidate generator appropriate to source language; authoritative project term sources. | A project with a non-English source can submit candidates without using the English capitalisation detector. |
| G9: multilingual reading | Workflow reads selected records across locales and reports missing translations, with optional ordered groups and annotations. | Catalogue adapter; grouping, ordering and speaker/usage annotations when available. | Read a checkout flow in screen order using the same reader that displays a t3 scene. No scene or hero vocabulary is required by the reader. |
| G10, T1: export and delivery | Workflow applies the agreed precedence, exports, then compares expected identities with identities present in host artefacts; reports losses even when the compiler exits successfully. | Catalogue writer, build invocation, artefact key reader and declared identity mapping. | A compiler exits zero but drops a selected key; the workflow reports incomplete delivery. |
| G11, F4, T2: formats | Reusable format support reads/writes catalogues and validates the declared message syntax. Workflow uses the same identity mapping for coverage. | Selected format implementation and runtime integration. | A plain CSV project needs no Lingui conventions; a normalised runtime key maps back to the expected source identity without a false loss. |
| Context pipeline, F1 | Workflow resolves context using an explicit precedence, retains manual decisions, reports origin and supports unresolved/deferred work. Engine consumes the resolved text. | Source facts, opaque domains and rules deriving context from product data. | A manually supplied explanation survives regenerated source context; a second project uses document sections instead of game tables. |
| L8: references between strings | Workflow checks declared label references against the corresponding locale values; heuristic discovery remains distinguishable from confirmed links. | Links between identities, or a classifier that proposes them; policy for permitted forms. | Two controls share the same source label: the supplied identity disambiguates the intended target instead of a text search choosing one. |
| L7, L4: grammatical/style checks | Common reporting and optional language checks with stated limitations. | Product voice, relevant entities/forms and supported language rules. | A detector can flag a suspected inflection issue without silently rewriting a name or becoming a universal engine rejection. |
| Review bookkeeping, R6 | Workflow records a decision against the text it reviewed, detects stale evidence and prepares guarded changes. | Reviewer decisions and reasons; optional importer for existing review documents. | A changed translation invalidates the applicability of an old approval; free-form advice is not treated as an approved replacement. |
| Agent skills and init | Skills compose these operations, gather missing decisions and explain results. Deterministic operations remain callable without a specific agent host. | Repository knowledge, permissions and host interaction facilities. | A configured project can inspect/check/apply an approved change without Claude-specific tools or an interactive interview. |

### Changes this implies for the current plan

These are proposed corrections to the boundary, not permission to port the scripts.

1. **Split ordered reading and review bookkeeping instead of leaving both wholly
   in t3.** `C/scripts/translation-scene.js:146` renders ordered groups across
   locales, whereas `:117` and `:135` interpret game speakers. The former is a
   reusable reading operation. Likewise `C/scripts/translation-review.js:118`
   checks whether review text is still current; the particular review decisions
   and t3 Markdown conventions are separate from that reusable requirement.
2. **Treat English candidate mining as a strategy.**
   `C/scripts/translation-terms.js:39` uses an English capitalisation regex and
   `:23` fixes the comparison locales. Candidate presentation and decisions
   generalise; that detector is not a language-independent default.
3. **Do not equate normalisation with safe migration.**
   `C/scripts/translation-rekey.js:29` removes all whitespace, and `:90` drops an
   old value when the destination is populated. G7 needs an identity/conflict
   contract before this behaviour can become shared. No full identity-storage
   migration is approved by identifying this prerequisite.
4. **Carry the reviewed expectation through to application.**
   `C/scripts/translation-edits.js:55` takes `from` from the current catalogue,
   rather than the value recorded in the review. The subsequent guard in
   `C/scripts/translation-apply.js:74` therefore detects changes after conversion,
   but not a stale review at conversion time. Importing a proposal must preserve
   its reviewed expectation or report that the expectation is unavailable.
5. **Keep providers out of Tradusco's internal write protocol.**
   `C/scripts/tradusco/sync-from-catalogs.js:85` and `:95` directly write progress
   and CSV. The general return path is an import operation; deciding writer
   coordination and recovery belongs with Tradusco's preservation contract.
6. **Do not make Node, gettext or an agent host prerequisites by inference.**
   Their current use establishes working t3 implementations, not a requirement
   for all integrations. Select packaging after defining which operations a
   configured standalone installation promises. Reuse existing parsers where
   they meet those operations; a second implementation is not required merely
   to make the project appear independent.

### Minimum boundary data to agree next

No schema is selected here. Operations need enough information to identify the
record and locale, distinguish source revisions, carry opaque domain/context
evidence, retain the value against which a correction was approved, and report
whether a change was proposed, persisted or exported. Optional ordered groups
and string-reference links serve reading and cross-string checks; a simple
catalogue does not have to invent them.

The next decision is preservation and authority: which inputs can replace which
values, what happens when the source changes, and how conflicts are resolved.
Until that is agreed, G2, G5, G7 and G10 have a shared unresolved prerequisite;
moving their files earlier would not resolve it.

## Preservation and authority proposal

Requested follow-up to the responsibility map. The owner agreed three rules on
2026-09-07: changed source is translated automatically without mandatory review;
unexplained changes from a known automatic value are presumed editorial and
outweigh automatic text; editorial text outweighs the glossary, with informational
disagreements only. These supersede the initial mandatory-review proposal and are
now recorded in the plan's R section. Other mechanics below remain proposals.
Nothing is implemented; storage layout, metadata schema and migration remain
undecided.

### Authority belongs to a decision, not a filename

The current paths implement different precedence rules:
`C/scripts/tradusco/export_translation_json.js:221` applies fixes and `:236`
then applies config overrides, while back-sync at
`C/scripts/tradusco/sync-from-catalogs.js:63` takes differing catalogue values.
Moving these scripts unchanged would preserve operation-dependent authority.

I recommend one resolved translation per identity, source revision and locale,
with its origin and applicable review evidence. That is a logical contract, not
a requirement for a new database or a single physical file. CSV, progress and
host catalogues must not silently compete to define that value.

| Input | Authority it supplies | What it must not silently do |
| --- | --- | --- |
| Project source snapshot | Current identities, source text and membership in the declared corpus. | Delete translations merely because an identity left the current selection or extraction was incomplete. |
| Explicitly accepted whole-string correction | Replacement for the specified cell and reviewed source/value, subject to structural validity. | Apply to a different source revision or overwrite an intervening edit. |
| Accepted glossary/context/style decision | Rules and evidence for the cells to which it applies. | Rewrite editorial text through export overlays. Glossary disagreement is informational, without mandatory review. |
| Model result | Candidate for the requested cell; it may become the working translation after applicable validation and write checks. | Gain human-review status by passing checks, being exported or being used as a reference. |
| Externally edited catalogue | A change from a known automatic value unexplained by recorded automatic writes is presumed editorial and takes precedence over automatic text. | Claim confirmed human review from that inference, or silently choose between competing editorial changes. |
| Exported or compiled artefact | Evidence of what was delivered and a baseline for detecting later host edits. | Become editorial authority solely because it is newer on disk. |

“Accepted” must name the relevant decision; agent output is not automatically
human-reviewed. A project can authorise a review process, but its provenance
must be reported truthfully. Recording approval without changing the text is a
valid operation too.

### Proposed default rules

1. **Protect editorial work for its source revision.** Ordinary translation
   fills eligible gaps; selected regeneration can replace unreviewed working
   values, but retains the previous value until the replacement is persisted.
   Confirmed or presumed editorial cells are protected while their source is
   unchanged. A changed source is translated automatically and its new result
   goes to the project after normal checks, without mandatory review. Preserve
   the old source/key, translation and editorial history. Ambiguous identity
   mapping remains a conflict case. A failed regeneration leaves the old value preserved and the
   requested replacement unresolved, not reported as successfully regenerated.
2. **Bind approval to what was reviewed.** Preserve identity, locale, source
   revision and expected translation when preparing a correction. A later change
   to any of these requires reconciliation. Do not manufacture the expected value
   from the current catalogue during review conversion. A missing expectation is
   reported and requires a fresh decision before replacement.
3. **Separate freshness from trust and membership.** Changing the source keeps
   the previous translation and review history, but removes the claim that it is
   approved for the new source; this does not block automatic new translation.
   A glossary change may produce an informational discrepancy for editorial text,
   not a requirement to reapprove it. Context/style invalidation policy remains
   undecided. Unrelated changes do not invalidate every approval. Retiring a source
   preserves its translations; reactivation checks revision and applicable rules
   before reuse. Unreviewed archived text is not worthless by definition.
4. **Preserve human text when rules disagree.** If a glossary rule conflicts with
   an accepted sentence, retain the sentence and show the applicable rule and
   evidence as information. Editorial text wins without a mandatory resolution
   step. An optional later editorial decision may change the sentence or rule.
   No substring replacement or precedence trick overrides editorial text.
   Structural invalidity still blocks the affected cell from successful delivery;
   preservation does not mean declaring invalid text safe.
5. **Treat host edits as changes to reconcile before export.** Compare the host
   value and internal value with their last agreed sync/export baseline. This
   distinguishes a host-only edit from two independent edits. Also compare with
   recorded automatic writes: unexplained changes from the known automatic
   value are presumed editorial and win over automatic results, even if both
   sides changed. Record that inference separately from confirmed review.
   Automatic export and correction scripts must participate in write provenance.
   Without a comparison baseline, first-import provenance remains unknown.
   Export must detect pending host edits rather than erase them.
6. **Missing is not deletion.** A missing host entry, empty value or absent source
   has a declared meaning from its adapter. It is not an implicit request to erase
   accepted work. An intentional empty translation is distinct from a missing
   translation when the selected format supports it. Retirement requires a
   complete authoritative source snapshot, not a filtered incremental selection.
7. **All writers honour the same contract.** Guarded apply, back-sync,
   regeneration and recovery check the expected revision at persistence time.
   As the initial concurrency policy, refuse overlapping writes to the same
   Tradusco project; do not build concurrent merge machinery first. External
   editors remain possible, so catalogue checks are still needed. Read-only
   commands may continue, provided they identify the snapshot they report.

### Sync and export decision table

Here `B` is the last agreed per-cell baseline, `T` the current Tradusco value and
`H` the host value. These cases assume the same identity and source revision,
valid input, and no conflicting review/protection metadata. Metadata-only
changes must also be reconciled; equal text does not erase review evidence.

| Observed values | Proposed outcome |
| --- | --- |
| `T = H` | Text already agrees; preserve and reconcile provenance. No text rewrite is needed. |
| `T = B`, `H != B` | Classify the host change against automatic-write history. Presumed or confirmed editorial text wins over automatic text; retain its actual provenance. |
| `H = B`, `T != B` | Internal-only edit: export after validating and checking that `H` still equals the expected value. |
| `T != B`, `H != B`, `T != H` | Apply provenance first: editorial wins over automatic. Competing editorial or otherwise unresolved changes remain conflicts; preserve both and the baseline. |
| No baseline and `T != H` | First-sync conflict: present both and require an explicit choice. File timestamps and directory names are insufficient evidence. |
| Missing entry or changed source revision | Use the declared missing/retirement or source-change rules; do not treat it as an ordinary value replacement. |

The baseline is required evidence for automatic two-way reconciliation. Its
physical representation is not selected here. An initial version without a
baseline can still work by reporting divergent values for explicit resolution;
it cannot honestly promise automatic identification of host-only changes.

### Recovery and acceptance examples

Persisting a translation and delivering it to the host are separate outcomes.
A crash between them must leave recoverable work, not a false report of complete
delivery. This requires a recoverable write procedure; this proposal does not
choose transactions, a journal or another storage mechanism. Refusing concurrent
writers alone does not solve interruption between writes.

Proposed offline acceptance cases:

- Accept a correction, export, sync and restart: text and review evidence survive.
- Prepare a correction against A, change the live text to B, then apply: report a
  conflict and preserve B. Already-applied text still requires checking whether
  the correction's review evidence matches the current source revision.
- Change known automatic host A to B without an automatic-write record and
  regenerate internal A to C: B wins as presumed editorial; retain C's history.
  Two competing editorial changes instead produce a conflict.
- Change an editorial source string: produce and export a new checked translation
  without a review gate; retain the old key/source, translation and review history.
- Change the source while an old model request is in flight: its result cannot
  mark the new source translated or replace a later correction.
- Interrupt after internal persistence but before export: report persisted but
  not delivered; recovery exports or reports a new host conflict without needing
  another translation call.
- Retire and restore an identity: preserve history, check freshness and review
  applicability; processing a three-row subset does not retire the rest.
- Introduce a conflicting glossary rule: preserve the editorial sentence and
  report the disagreement informationally, without blocking or demanding approval.

### Consequences for the plan

G2/G5/G7/G10 and W8 need these rules before their acceptance criteria can close.
R4/R6 must distinguish automatic replacement protection, review evidence and
freshness. R5 cannot simply inherit the prototype's trust policy. L9 should
remove competing editorial authorities, not forbid a glossary value from also
appearing in an accepted sentence. O2/O3/W3 must distinguish preserved old text,
successful replacement and successful delivery.

The three agreed rules above resolve source-change handling, inferred editorial
priority and glossary priority. Conflict/recovery mechanics beyond those rules,
exact metadata/storage and reference-example reuse policy remain
separate decisions. In particular, this proposal does not settle R2's quality
experiment or require reviewed-only examples.

## Completion and recovery: agreed direction

**Owner agreed the presented scenarios on 2026-09-07:** retain successful cells,
retry only unresolved work within a budget, recover delivery without retranslating
persisted results, resume from persisted work after interruption, and export the
ready subset immediately with an explicit partial outcome. Details below beyond
those scenarios remain design proposals; no live run or retry values are authorised.

The existing wrapper calls translation before its export/recovery decision
(`C/scripts/tradusco/run.js:529`). Changing the engine to return nonzero on partial
failure without updating that caller would stop the intended recovery path.
Outcome semantics therefore belong to O2/O3/W3 and G1 together.

### Proposed behaviour

| Scenario | Recommended behaviour | What the operator sees |
| --- | --- | --- |
| 97 of 100 requested cells persist successfully; three exhaust retries | Retain the 97. Automatically retry only unresolved eligible cells within the configured run budget. Once exhausted, end partial; a later resume selects the remaining three after reconciling current source and editorial changes. | Requested, satisfied, unresolved and protected/superseded cells, with reasons. No success claim based merely on nonempty old values. |
| Translation persists, then export or build fails | Resume the failed deterministic delivery steps using saved translations. Revalidate source and host changes before writing; do not call a model again just because delivery failed. | Translation complete; delivery incomplete, naming the failing stage and affected scope. |
| Process is interrupted during a request | Retain persisted work. Reconcile any recoverable result before retrying. A dispatched request without a recoverable persisted result remains unresolved. | Interrupted run and exact remaining selection; provider execution/cost may be unknown for the interrupted request. |
| Source or editorial text changes during a run | Reconcile before accepting the old result. A result for an old source cannot satisfy its replacement; editorial changes for the same source win over automatic text. | Superseded work distinguished from model failure. New source work belongs to a refreshed selection, rather than silently changing the running scope. |
| Some translations are ready while others failed | By default, export valid resolved values for the selected scope without deleting unresolved or unrelated host entries. Mark the workflow partial until required delivery coverage is satisfied. | Ready values available in project files, with explicit gaps. Local export does not imply permission to publish or release. |

Partial export is the agreed default: make useful completed work available
immediately. An integration whose catalogue
cannot represent a partial update may reject the export as a whole; Tradusco must
not pretend that such an adapter delivered the successful subset. Existing stale
host values, if preserved, are not counted as current successful translations.
Host fallback and release policy remain project responsibilities.

### Meaning of completion

- Fix the requested identities, source revisions, locales and requested stages
  at run start. Reconcile later changes explicitly rather than grow the run forever.
- A model response is not yet success: it must pass applicable checks and be
  persisted against the still-current expected revision. Replacement that produces
  the same text can succeed; a textual diff is not required as evidence.
- For a translation-only operation, completion means the requested eligible cells
  are satisfied and persisted. For an export/build workflow, the requested delivery
  checks must also pass. A deliberately skipped stage is reported as not checked.
- A recovered intermediate failure does not make the final run fail. Remaining
  failures, unresolved conflicts or superseded requested work make it incomplete.
  Editorial protection is reported separately and never claimed as regeneration.
- Return a structured summary plus a nonzero outcome for incomplete work; callers
  distinguish partial, interrupted and fatal outcomes so partial results can still
  reach deterministic recovery. Exact numeric exit codes are not selected here.

### Bounded continuation

Retries, format/model fallback and split children consume the declared logical
work budget; splitting must not reset it. No timeout or attempt values are chosen
in this block. A new resume is a visible new invocation with its declared budget,
linked to the prior run; do not implement endless automatic resume cycles.
Authentication/configuration failures and persistence failures stop the affected
run, rather than be treated as defects that smaller translation batches will fix.

After a hard interruption, a new request may be necessary if no usable result
survived. Do not promise exactly-once model execution or zero duplicate billing.
For known persisted results, however, a delivery retry must require no model call.

Offline acceptance should exercise partial locale failure, equal-text successful
regeneration, interruption between persistence and export, an intervening editorial
edit, and a compiler that exits zero but drops a key. These are proposed checks,
not executed tests. Their implementation follows only after document agreement
and separate implementation authorisation.

## Input readiness and checks: agreed direction

**Owner agreed the presented policy on 2026-09-07:** automatic deterministic
preparation, nonblocking missing guidance, explicit provider failures, honestly
labelled machine references, optional reference-first ordering, and bounded repair
for technical failures rather than heuristic warnings. Reflected in W2/W5/W7 and
L1/L5 of the plan. Command names, schema and implementation remain undecided.

1. **Prepare available inputs automatically for the selected changes.** Run
   configured deterministic glossary/context providers and resolve their outputs
   before translation. Do not require a human to approve each run or invent
   missing context. Inspection uses the same selection and resolution without
   persisting generated artefacts or calling a model.
2. **Missing guidance normally does not block translation.** Report missing
   context, target glossary forms and references per selected cell, and proceed
   with the usable inputs. Absence of a matching glossary entry is not a defect
   by itself. Distinguish absent optional guidance from a configured provider that
   failed: a provider failure must be explicit, and blocks the dependent stage
   unless the project has declared that provider optional. Projects may declare
   specific required inputs; do not hardcode t3 requirements into the default.
3. **Use reference provenance honestly.** A missing reference cell is omitted.
   References/examples for a different source revision are not current guidance.
   For current source revisions, do not add a blanket reviewed-only restriction
   before R2's quality experiment: permit machine and presumed-editorial material
   under the selected policy, labelled with its actual provenance. Confirmed
   review, presumed editorial origin, machine origin and unknown legacy origin
   must not all be described as reviewed. The original source remains authoritative.
4. **Reference-first translation is optional.** A project may choose to translate
   reference locales first, including the extra calls in its run plan and budget.
   It is not a mandatory bootstrap dependency. Do not pause for human review
   between phases merely to make fresh machine references look reviewed. Gate the
   actual input requirements, not whether an arbitrary phase order was followed.
5. **Separate technical failures from heuristic warnings.** Malformed output,
   missing required placeholders and invalid declared format prevent the affected
   result from succeeding and can trigger bounded repair of automatic output.
   Identical source/target text, unusual script, length or typography are normally
   warnings. Do not spend on repair solely because a heuristic fired. Project
   policy may explicitly promote a supported check; editorial priority and
   protection still apply. Glossary disagreement with editorial text remains
   informational under the already agreed rule.

Acceptance examples for the agreed policy (not executed tests):

| Situation | Default outcome |
| --- | --- |
| Three new strings have no references yet | Translate from source with available context/glossary; no preparatory review required. |
| A configured context generator crashes | Report the failure; stop dependent translation unless it was declared optional. Do not silently treat the crash as ordinary zero coverage. |
| A current reference was machine-translated | It may be used under the reference policy, identified as machine output; no human-reviewed claim. |
| A translation equals a product name in the source | Warn if the configured detector flags it; do not automatically rewrite or block it. |
| A model result drops a required placeholder | Retry the affected automatic result within budget; retain successful cells and report any remaining failure. |

Input inspection reports exact selection and assembled-input measurements
separately from estimated output cost, cache behaviour and future split counts.
A coverage percentage is not a quality score. Nothing in this block claims that
unreviewed examples improve quality; the proposal preserves their possible use
while R2 remains an explicitly separate experiment.

## Proposed documentation and review process

Keep the plan as the current design; keep this review as findings and proposals,
and the response as a handoff. Do not start a new specification for every comment.

Three passes have different exit conditions:

1. **Facts:** a claim has a reproducible command or a code reference, a dated
   measurement has a unit and corpus, and historical reports are distinguished
   from current behaviour. A causal hypothesis is not promoted by repetition.
2. **Decisions:** agree on audience, the first user scenarios, preservation and
   trust rules, failure behaviour, and what is intentionally deferred. For each
   contested point record the options, recommendation and the owner's decision.
3. **Execution readiness:** every selected item has observable acceptance,
   prerequisites, compatibility/rollback expectations and an evidence method.
   Separate offline checks from paid experiments with their own budgets.

Stop reviewing a selected delivery when it has no unresolved decision that would
change its behaviour, data contract or recovery path. Nice-to-have ideas remain
deferred instead of endlessly expanding the delivery. Contradictions and broken
invariants still block agreement; wording preferences do not.

Document agreement and implementation authorisation are distinct events. A later
"continue reviewing" instruction is not permission to code. Changes to the
accepted design reopen only the affected decision and its dependants. Translation
of the remaining Russian documents, history rewriting and pushing remain subject
to the user's existing separate sequencing constraints.

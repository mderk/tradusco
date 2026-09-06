# Response to the tooling-plan review (2026-09-07)

What was done with the review of `REWORK_PLAN_TOOLING.md`, what changed in the
plan as a result, and what the next person picks up. Read this before
`REWORK_PLAN_TOOLING.md`: it says which parts of that document are new and which
were already there and are now wrong to quote from memory.

Every finding was accepted. Nothing was rejected as inapplicable, and no finding
was left unaddressed. Three of them contradicted claims the plan made with a
`file:line` reference, so those were re-checked against the code directly before
the plan was edited; all three held.

## Verified independently

- `ChatOpenAI` as the driver builds it reports `request_timeout None` and
  `max_retries None`; the client underneath it reports `timeout None` and
  `max_retries 2`. The plan's earlier claim of a 600-second default was wrong —
  there is no bound at all, and the SDK's retries are nested inside
  `BaseDriver.py:99`.
- `_Rule.spans` (`lib/envelope.py:42`) handles `skip` and `exact` and sends every
  other value to the regex branch. `_Rule("Cat", {"mode": "typo"}, 0).spans("A Cat
  sleeps")` matches, so a typo widens a rule rather than disabling it.
- `.claude/skills/t3-translation-context/scripts/lib/context.js` reads
  `conf_src/heroes/heroes.csv` and walks eight fixed ability slots per hero
  (`:104`), and `scripts/lib/po.js:47` classifies this game's catalogs as
  narrative. The plan's "knows nothing about t3" was wrong.
- The review's counts reproduce exactly against the current t3 corpus: 939 of
  1032 glossary entries in `exact` mode, 3559 corpus rows against 4455 catalog
  keys (896 out of corpus), 3629 progress keys of which 70 are dead, and 16
  quote-check findings.

## What changed in the plan

Corrections to items that were wrong:

- **O4** — restated around an unbounded wait rather than a large default, with
  the nested SDK retries named. The default value is explicitly undecided: it has
  to be overridable per driver and per model, and picked from O3's distribution
  of successful request durations.
- **C5** — demoted to a proposal. One unreproducible run does not establish an
  exhausted output budget, `parse_error` also covers a missing language block and
  model-level failures, and the same heuristic was already built and removed
  (`A6_BATCH_TASK.md:3`). The item now states the comparison that would settle it.
- **C4** — keeps its policy, loses its proof. The "eight requests against fifteen"
  sketch models a one-shot split into fives; the engine halves recursively
  (`TranslationProject.py:546`, `:603`) with driver retries underneath, which
  comes to roughly seventeen calls per 150 rows in the same scenario.
- **F2** — the failure direction was backwards; a bad mode widens a rule to stem
  matching.
- **L6** — the 91%-inert reading was a misreading of the same number. `exact`
  entries are mostly whole-phrase UI labels: 974 of 980 effective `exact` rules
  match at least one corpus row. The item now names four distinct diagnoses that
  the coverage report has to separate.
- **L8** — sixteen string-and-locale pairs, one label accounting for ten of them.
  The class is not particular to games, and quoting is only the first heuristic:
  the general form has the project supply the link between two keys.
- **R5** — the scale figure was from before pruning ran, and t3 already has a
  prototype (`prune-progress.js`), so this is a port with a policy, not a design.
- **Three layers** — the layer-2/3 split in the t3 context skill is not clean.
  Three separations have to happen before G1, G6 and G11 can move: the game's
  data access, its content classification, and gettext reading.
- **G5** (in "Still open") — writing catalog and fixes file in one call is not
  durability. The catalog is written first (`translation-apply.js:89`), and a
  re-run does not repair a half-written edit because the catalog value already
  matches and the edit counts as `already` (`:70`). Also `from` is honoured only
  when supplied (`:74`).

Items folded into others rather than kept separate:

- **W6, W7** are now behaviour of the orchestrator: built inside G1, reported
  through W5, sequenced through W1 and R1. W6 contributes an acceptance test —
  three new strings, one command, three decisions. W7 additionally carries the
  trust question, since a reference locale translated in phase one is not
  reviewed and must not be presented to the model as if it were.
- **L9** no longer proposes merging the glossary with the fixes layer; they
  answer different questions. What is duplicated is canon living in the override
  layer, and the check is against applicable rules (L2), not equality of two
  dictionary values.
- **L2** gains the reason it cannot be ported as-is: the t3 conformance check
  decides applicability differently from the engine (`translation-lint.js:34`,
  `:92`, `:115` against `lib/envelope.py:40`, `:57`). A shared rule contract with
  one set of fixtures for both languages comes before L5's feedback loop.
- **L7**, **C3** demoted to proposals — neither the inflection detector's error
  rate nor the cache saving has been measured.
- **R2** — the mechanism is confirmed and sized (1206 of 3564 phrases get such
  examples for `uk`); the harm is not, and that comparison decides between the
  two cures.

New items:

- **L10** — an entry that fires can still lack a form for the target language.
  Report missing forms per entry per language, and an import path for confirmed
  canon.
- **R6** — review status per cell, set by the project. R5's restore policy, R2's
  example selection and W7's reference phase all depend on it.
- **W8** — the engine writes progress under a lock and atomically
  (`lib/storage/filesystem.py:206`), the back-sync prototype writes the same
  files directly (`sync-from-catalogs.js:85`). Settle the write protocol before
  G2 ships.
- **F4/T1/T2** — a format adapter rather than configurable regexes; key reading
  and normalisation are format work too. T2's wording was wrong about the
  runtime: it applies the same substitution rather than reversing it
  (`client/src/utils/l10n.js:9` against `scripts/lib/dotted-args.js:12`), which
  is why it must be one function.
- **W3** — a failure log is a log of attempts; the report on it has to resolve
  each key-and-language pair to a final outcome.

The order of work was rebuilt: O4 alone first, then O3 and the O-group, then
C5's measurement, W5/W2, F5's contract before the formats, F4, F1, L2's contract
before L6 and L1, T1/T2, then G1 and G2 together with W6/W7/W8 inside them, F5's
migration, and the skills last.

## Repository state

Branch `main`, 23 commits ahead of `origin/main`, nothing pushed. The relevant
commits from this session, oldest first:

- `ca21059` — the one Russian commit message, rewritten in English (amended,
  message only).
- `c0113f8` — the A1, A2-A5 and A6 task briefs, previously untracked.
- `602fc67` — status blockquotes on four plans, corpus note on `BATCH_BASELINES.md`.
- `73d4e67` — `README.md` and `INTEGRATION_GUIDE.md`: multi-language `--lang`,
  `--reference-langs`, `--batch-max-input-tokens`.
- `5f6cace` — `WORKFLOW_NOTES.md`: shipped separated from open.
- `b9c5c85` — `REWORK_PLAN_TOOLING.md` and `REVIEW_TASK_TOOLING.md`, the
  pre-review state.
- `0844cd9` — this review folded in.

`REVIEW_TASK_TOOLING.md` also gained a correction: the dry run it recommends
regenerates glossary and context artefacts, so it now passes `--project-dir` at a
throwaway path.

## Constraints in force

- **Do not push.** Six documents are still in Russian — `WORKFLOW_NOTES.md`,
  `REVIEW_SKILL_SPEC.md`, `REWORK_PLAN_MULTILANG.md`, `A1_MULTILANG_TASK.md`,
  `A2_A5_ENVELOPE_TASK.md`, `A6_BATCH_TASK.md` — and this is a public
  repository. All six entered history in unpushed commits, and `origin/main`
  contains no Russian at all, so the episode can still be erased for free. The
  plan: keep working, translate once the documents are settled, then strip those
  six paths from history with `git filter-repo --invert-paths` and re-add the
  English versions as one commit. Pushing before that makes the rewrite
  expensive.
- **Write new documents in English.** Only the six above are exempt, and only
  until they are translated.
- **Nothing that spends money** without explicit intent: `translate.py`,
  `yarn tradusco*`, `run-all.js` without `--skip-translate`. Keys live in
  `.env.tradusco` and must not reach any output.
- **Data files are edited by scripts, not by hand:** `conf_src/**`,
  `server/conf/**`, `.tradusco/booty/*/progress.json`, `.tradusco/baseline`,
  `.tradusco/after-150`.
- The t3 working tree carries other processes' uncommitted files. Stage only
  paths you touched.

## What is next

Nothing in the plan is implemented yet, and the review did not change that. The
first three things it now says to do:

1. **O4**, on its own. A few lines on the driver plus the decision about where
   the timeout value comes from. Everything below is measured on runs, and a run
   that hangs cannot currently be told from one that works.
2. **O3**, the run log, then O1, O2, O2a on top of it.
3. **Measure R2**, which does not wait on either. `EnvelopeBuilder` is pure, so
   building an envelope from the current CSV with `--regenerate` semantics and
   reading what comes out answers it without calling a model. If examples really
   do anchor regeneration to the text it replaces, that changes the quality of
   everything translated from scratch.

Also open and unscheduled: sixteen quote-check findings in t3 left unfixed by a
release-time decision (`Find Match` in ten locales, `Settings` in es-la and uk,
`Check status` in it and ro).

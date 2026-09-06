# Task: review of the tradusco rework plan

Under review: `REWORK_PLAN_TOOLING.md` — the plan for which parts of the
tradusco integration in the t3 project move into tradusco itself, and what
tradusco needs for that. The plan is open; nothing in it has been implemented.

Your job is not to agree and not to rewrite, but to check: do the claims match
the code, is the problem invented, is it placed in the right layer, and is
anything important missing. The items added last (`L8`, `L9`, `W6`, `W7`, `C5`,
`O4`) were written from a single day of work, and that is the weakest part of
the document: one observation is easily mistaken for a pattern.

Change nothing. Not in tradusco, not in t3. The result is a text with findings.

## Where things are

Two trees, read both:

- `/Users/max/Documents/projects/translator/ai_translator` — tradusco itself.
  Engine: `lib/TranslationProject.py`, `lib/TranslationTool.py`,
  `lib/envelope.py`, `lib/failure_reporting.py`, `lib/llm/*/`,
  `prompts/translation.txt`. Entry points: `translate.py`,
  `sync_project_from_csv.py`, `audit_translations.py`, `po_status.py`.
- `/Users/max/Documents/projects/t3/client` — the integration.
  `scripts/tradusco/` (orchestrator, preflight, export, coverage check),
  `scripts/translation-*.js` (about thirty scripts), `scripts/lib/`,
  `.claude/skills/t3-translation-*` (two skills: glossary and context),
  `tradusco.config.json`.

Neighbouring documents, without which the plan reads wrong: `WORKFLOW_NOTES.md`
(general conclusions, some already implemented), `BATCH_BASELINES.md`
(measurements and the batching policy in force), `GLOSSARY.md` (glossary format
and selection rules), `README.md`, `INTEGRATION_GUIDE.md`. The closed plans
`REWORK_PLAN.md` and `REWORK_PLAN_MULTILANG.md` — so you do not re-propose what
has already been cancelled; `A6_BATCH_TASK.md` is explicitly marked "cancelled,
do not implement".

## Order of context gathering

Thirty minutes, no more. The goal is to understand the design, not to proofread
the code.

1. `REWORK_PLAN_TOOLING.md` in full, once, without checking anything. Note the
   "Three layers" section and "Non-goals" — that is the criterion by which
   items are accepted or rejected.
2. `WORKFLOW_NOTES.md` and `BATCH_BASELINES.md` — what is already decided and
   why.
3. A dry run of one language, to see the steps live. It costs no money, but the
   skipped steps are not the only ones that write: this still regenerates the
   glossary and context artefacts, so point it at a throwaway project directory
   rather than the real one:

       cd /Users/max/Documents/projects/t3/client
       node scripts/tradusco/run.js --lang ru --project-dir /tmp/tradusco-review \
         --skip-extract --skip-sync --skip-translate --skip-apply --skip-build \
         --skip-export-json

4. `lib/envelope.py` — request assembly: context, glossary, reference
   languages, slicing. Most of the plan's items pass through it.
5. `scripts/tradusco/run.js` and `run-all.js` — what the plan proposes to move
   into tradusco as `G1`.

## What exactly to check

**Facts.** An item with a `file:line` reference makes a checkable claim — check
it. Lines may have moved; what matters is whether what is described is at that
place. An item without a reference is a proposal, not a finding, and the
requirement for it is different: is it grounded in anything at all.

**Numbers.** In the plan they serve as evidence that a problem is real and set
its scale. Check reproducibility for at least three, including:

- "934 of 1027 glossary entries in `exact` mode" (`L6`) — a `node -e` over
  `translation_glossary.json` in t3;
- "the catalog is 896 keys wider than the corpus" and "the corpus is 3559 rows"
  — `locale_src/<lang>/translation.json` against
  `.tradusco/booty/translations.csv`;
- "sixteen long-standing places where quotes disagree, one of them in nine
  locales" (`L8`) — `node scripts/translation-quote-check.js`.

If a number does not hold up, that is a finding, even when the conclusion
remains correct.

**Layer.** For each item: engine, workflow, or project map. The test is single —
could a second project with a different stack adopt this without patching
tradusco. Pay particular attention to whether an item smuggles knowledge of
gettext, lingui, placeholder syntax, domain file names or the game into the
engine.

**Duplicates and contradictions.** The plan was written in several passes. Look
for items that say the same thing in different words, and items that get in each
other's way. A known candidate: `C4` defends the current slicing policy, `C5`
demands that batch size be counted differently; whether they can coexist is a
question for you.

**Order of work.** The "Order of work" section at the end. Check the
dependencies: is an item placed before something it cannot be done without, and
the other way round.

## Questions that need an answer

1. `L8` (one string quoted inside another) — a real class of defects or a
   special case of t3? Look at other projects you know: do instructions that
   name a button occur outside this game.
2. `L9` (the fixes layer against the glossary) — is the cure right. Perhaps the
   two mechanisms should not both exist, and one of them is redundant.
3. `W6` and `W7` (the incremental cycle and the reference-language order) — are
   they smeared across three other items (`W1`, `W5`, `G1`), and would it be
   simpler to fold them in there.
4. `C5` — is the move from "batch size in rows" to "batch size by output volume"
   justified by a single measurement. What has to be measured for this to become
   a decision.
5. `O4` — the driver timeout. What default value, and should it be the same for
   all drivers.
6. What the plan is missing. This is the most valuable answer. Look at what the
   t3 integration does a lot of and by hand, but which did not make it into the
   plan.

## Boundaries

- Run nothing that spends money: `translate.py`, `yarn tradusco*`, `run-all.js`
  without `--skip-translate`. The keys live in `.env.tradusco` and must not
  reach any output.
- Do not touch `conf_src/**`, `server/conf/**`,
  `.tradusco/booty/*/progress.json`, `.tradusco/baseline`,
  `.tradusco/after-150`. These are data, edited only by scripts.
- Do not commit. The t3 working tree sometimes holds other people's
  uncommitted files.

## What to return

A table of findings, the most substantial first:

| code | item | what is wrong | evidence | recommendation |

`code` — yours, running. `evidence` — `file:line` or a command with its output,
not a retelling. `recommendation` — one of: keep as is, rephrase (how), demote
to a proposal, drop as a duplicate, move to another layer, raise in the order of
work.

Separately, as lists and without a table:

- items you checked and confirmed — briefly, so the coverage is visible;
- omissions: what the plan lacks and should have, with the same requirement of
  evidence;
- questions you had too little data for, and what exactly needs measuring.

Do not rewrite the document itself and do not propose wording changes for the
sake of style.

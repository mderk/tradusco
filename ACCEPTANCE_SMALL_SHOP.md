# Repository acceptance scenario: a small shop

**Specification for implementation with the rework; not an existing test.** Keep
the example data, adapters and acceptance checks in the Tradusco repository.
Each execution copies fixtures into a fresh temporary project. It must require
neither a t3 checkout nor its data, credentials, configuration or agent skills.
The example also demonstrates how another project integrates the common workflow.

This is the expanded specification of the nine-step acceptance round trip in
[`REWORK_PLAN_TOOLING.md`](REWORK_PLAN_TOOLING.md#worked-round-trip-for-review).
The step references below identify which part of that concise contract each event
exercises; events without a direct counterpart extend its fault and identity
coverage.

Use English source and French, German and Japanese targets. Initial data:

| Owner | Logical identity | Source | Initial condition |
| --- | --- | --- | --- |
| Interface | `checkout.title` | Checkout | Existing editorial translation, with explicit evidence for its source revision. |
| Interface | `checkout.pay` | Pay {amount} | Missing target values; declared placeholder validation applies. |
| Interface | `common.back` | Back | Existing machine values; an existing context rule describes navigation. |
| Products | `42.name` | Travel Pack | Existing machine values; Pack is a glossary candidate. |
| Products | `42.description` | Everything for a short trip | Existing machine values; source will change during the scenario. |
| Host only | `legacy.notice` | Legacy notice | Present in host targets but never managed by either extraction. |

Run the same events through two configurations: a CSV phrase-table integration
and a JSON catalogue integration with explicit IDs. Keep logical identities,
source text and expected outcomes equivalent; change the readers/writers and
configuration. Both expose separate interface/product ownership scopes. The
test must use the normal workflow entry points for selection, preparation,
application, export and recovery, rather than implement a second orchestrator.
Concrete fixture layout and command spelling follow the agreed integration API.

Two execution modes use this example:

- **Default offline acceptance.** Controlled model responses and injected failures
  exercise the actual engine/workflow. Only the external model boundary and
  intentional host failures are substituted; do not mock away preservation,
  persistence or reconciliation. Count model invocations and record which cells
  and source revisions they request.
- **Explicit live API integration check.** The same successful round trip uses a
  selected real provider/model, keys from the environment and a small declared
  request/token/time budget. It is opt-in and never falls back to paid execution
  from an offline test. Report provider usage and actual/estimated cost separately;
  a local timeout is not proof that provider billing stopped. Provider, initial
  limits and any spending ceiling must be selected before a live run. Do not
  depend on a real model spontaneously producing the failure fixtures.

## Events and observable acceptance results

Use the ordinary fixture for the successive events below. Fault cases branch
from its named checkpoints so one failed case does not contaminate another.
Observe persisted state, exported artefacts, summaries and the next invocation;
avoid exact assertions on an internal storage layout that has not been selected.

| Round-trip step | Event | Expected result |
| --- | --- | --- |
| 1 | Connect the fixture twice | Import preserves existing values and known provenance. Reconnection is idempotent. `legacy.notice` stays unmanaged. A separate divergent-legacy variant reports an unresolved import instead of inventing authorship. |
| 3 | Prepare terminology/context | Candidate evidence for Pack is presented. A recorded decision supplies its canon; an existing rule supplies Back's context. An unrelated rejected candidate is not proposed again on unchanged evidence; a deferred candidate remains resumable. Offline answers are fixtures, not a mandatory interactive approval screen. |
| 3 | Inspect without executing | Selection, pending decisions and missing guidance are visible. Project files remain unchanged and model-call count is zero. |
| 2 | Add `checkout.receipt` = Email receipt and change the product description | Select missing Pay cells, the new receipt and changed description for the three locales. Preserve prior source/translation history; do not regenerate Checkout, Back or the product name merely because extraction ran. Missing optional receipt context does not block translation. |
| 4–5 | Return one technical failure | The Japanese Pay result omits `{amount}`. Retain other valid cells, retry only unresolved eligible cells and export ready values. Exhausting the configured retry budget leaves an explicit partial result. A later resume does not translate successful cells again. |
| 7 | Apply a supported editorial correction | Change the French product name through the ordinary guarded operation. Record its origin at application time; repeating it has no additional effect. |
| 6 | Edit outside Tradusco | Change the exported German product name after a known automatic export. The next reconciliation treats its unexplained change as presumed editorial, distinct from confirmed review. It survives same-source regeneration/export. A competing-editorial variant reports a conflict. |
| 9 | Change Pack's canon | Preview reports affected input and eligible regeneration scope. French/German editorial names remain protected; the Japanese machine name can be regenerated. Unrelated cells are excluded. Record an explicit selection if dependency evidence is unavailable; do not claim inferred selection from missing history. |
| 7, 9 | Change the rule producing Back's context | Preview includes old/new effective values and affected rows. A manual-context variant is preserved; a no-statement variant removes only attributable obsolete generated context and resolves any fallback. Rule/input changes between preview and apply invalidate that preview. |
| 8 | Change Checkout's source to Secure checkout | Produce a new translation after ordinary checks without an editorial approval gate. Keep the old source and its editorial translation in history. |
| 5 | Fail the host build after translation persistence | Report translated but not delivered. Retry delivery from stored values with zero additional model calls. Also test a build that exits zero but omits a key: coverage must catch it. |
| Extension | Extract interface only, then remove and restore a product in complete product snapshots | Partial participation does not retire product records. Complete removal affects only previously managed product identities; history survives. `legacy.notice` is preserved throughout. Reactivation follows the specified revision/provenance policy. |
| 5 | Repeat completed work | No new model calls or effective value changes absent a new source/guidance change or an explicit regeneration request. Summaries do not count preserved stale text as a successful replacement. |

The live variant checks structure, declared placeholders, saved/exported coverage
and broad unambiguous glossary constraints rather than a single exact translation.
Editorial values are compared exactly because their preservation is the contract.
Do not equate a technically valid response with verified meaning or naturalness;
semantic quality evaluation remains separate from workflow acceptance.

## Fault checkpoints and limits of the example

Offline variants inject missing locale blocks, timeouts and nested retry/fallback
failures to verify the declared budget. At each boundary in the chosen persistence
procedure, terminate the operation and restart: no accepted edit may disappear,
no undelivered result may be reported delivered, and recoverable saved translations
must not be requested again. Inject an editorial or source change while a response
is pending to check that stale model output cannot overwrite it. Where no response
survived, a new request is permitted; exactly-once provider execution is not promised.

Add small identity variants to this same fixture: two equal source texts with
different meanings/IDs, colliding local IDs from different owners, and ambiguous
whitespace rekeying. These verify identity and ownership contracts without
inventing a second large example project.

Passing requires both adapters to satisfy the same observable contract and every
declared fault variant to reach its expected recovery state. Live success cannot
replace offline fault coverage. This example is the first workflow acceptance
baseline, not proof of every format, language, integration or translation quality.
Implement it alongside the selected vertical slice; the document does not imply
that currently missing APIs or unresolved migration policies already exist.

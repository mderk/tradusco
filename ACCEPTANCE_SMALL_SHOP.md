# Repository acceptance scenario: a small shop

**Acceptance specification partially covered by the repository test suite.** Keep
the example data, adapters and acceptance checks in the Tradusco repository.
Each execution copies fixtures into a fresh temporary project. It must require
no external checkout, data, credentials, configuration or agent skills. The
example also demonstrates how another project integrates the common workflow.

Use English source and French, German and Japanese targets. Initial data:

| Record | Source | Initial condition |
| --- | --- | --- |
| `checkout.title` | Checkout | Existing editorial translation. |
| `checkout.pay` | Pay {amount} | Missing target values; declared placeholder validation applies. |
| `common.back` | Back | Existing machine values; an existing context rule describes navigation. |
| `42.name` | Travel Pack | Existing machine values; Pack is a glossary candidate. |
| `42.description` | Everything for a short trip | Existing machine values; source will change during the scenario. |
| `legacy.notice` | Legacy notice | Present in host targets but never managed by the extraction. |

The completed acceptance suite must run these events through one CSV phrase-table integration
using the current source-as-key convention. The test must use the normal workflow
entry points for selection, preparation, application, export and recovery rather
than implement a second orchestrator. Add another adapter only when a real second
project requires it.

Two execution modes use this example:

- **Default offline acceptance.** Controlled model responses and injected failures
  exercise the actual engine/workflow. Only the external model boundary and
  intentional host failures are substituted; do not mock away preservation,
  persistence or reconciliation. Count model invocations and record which source
  keys they request.
- **Explicit live API integration check.** The same successful round trip uses a
  selected real provider/model, keys from the environment and a small declared
  request/token/time budget. It is opt-in and never falls back to paid execution
  from an offline test. Report provider usage and actual/estimated cost separately;
  a local timeout is not proof that provider billing stopped. Provider, initial
  limits and any spending ceiling must be selected before a live run. Do not
  depend on a real model spontaneously producing the failure fixtures.

The current tests cover the parts separately. `tests/test_reference_host.py`
runs the reference host through preparation, explicit back-sync and delivery.
`tests/test_acceptance_small_shop.py` covers translation, isolated technical
failure, resume, source change, explicit regeneration scope, editorial
preservation and the opt-in Gemini path. `tests/test_glossary_tool.py` and
`tests/test_context_tool.py` cover decision preparation. `tests/test_review_tool.py`,
`tests/test_delivery_tools.py` and `tests/test_run_tool.py` cover guarded review,
partial export, the project lock, failed-build resume and missing artefact keys.
The successive events below are not yet one end-to-end executable scenario.

## Events and observable acceptance results

Use the ordinary fixture for the successive events below. Fault cases branch
from its named checkpoints so one failed case does not contaminate another.
Observe persisted state, exported artefacts, summaries and the next invocation;
avoid exact assertions on an internal storage layout that has not been selected.

| Event | Expected result |
| --- | --- |
| Connect the fixture twice | Import preserves existing values. Reconnection is idempotent and `legacy.notice` stays unmanaged. |
| Prepare terminology/context | Candidate evidence for Pack is presented. A recorded decision supplies its canon; an existing rule supplies Back's context. An unrelated rejected candidate is not proposed again on unchanged evidence; a deferred candidate remains resumable. Offline answers are fixtures, not a mandatory interactive approval screen. |
| Inspect without executing | Selection, pending decisions and missing guidance are visible. Project files remain unchanged and model-call count is zero. |
| Add `checkout.receipt` = Email receipt and change the product description | Select missing Pay cells, the new receipt and changed description for the three locales. Preserve prior source/translation history; do not regenerate Checkout, Back or the product name merely because extraction ran. Missing optional receipt context does not block translation. |
| Return one technical failure | The Japanese Pay result omits `{amount}`. Retain other valid cells, retry only unresolved eligible cells and export ready values. Exhausting the configured retry budget leaves an explicit partial result. A later resume does not translate successful cells again. |
| Apply a supported editorial correction | Change the French product name through the ordinary guarded operation. Record its origin at application time; repeating it has no additional effect. |
| Change Pack's canon | Preview reports affected input and eligible regeneration scope. French/German editorial names remain protected; the Japanese machine name can be regenerated. Unrelated cells are excluded. Record an explicit selection if dependency evidence is unavailable; do not claim inferred selection from missing history. |
| Change Checkout's source to Secure checkout | Produce a new translation after ordinary checks without an editorial approval gate. Keep the old source and its editorial translation in history. |
| Fail the host build after translation persistence | Report translated but not delivered. Retry delivery from stored values with zero additional model calls. Also test a build that exits zero but omits a key: coverage must catch it. |
| Repeat completed work | No new model calls or effective value changes absent a new source/guidance change or an explicit regeneration request. Summaries do not count preserved stale text as a successful replacement. |

The live variant checks structure, declared placeholders, saved/exported coverage
and broad unambiguous glossary constraints rather than a single exact translation.
Editorial values are compared exactly because their preservation is the contract.
Do not equate a technically valid response with verified meaning or naturalness;
semantic quality evaluation remains separate from workflow acceptance.

## Fault checkpoints and limits of the example

Offline variants inject missing locale blocks and timeouts to verify bounded retry.
The existing progress-before-CSV boundary is restarted to verify that saved
translations are not requested again. Add further failure fixtures only for write
boundaries introduced by the transferred operations.

Passing requires the selected adapter to satisfy the observable contract and every
declared fault variant to reach its expected recovery state. Live success cannot
replace offline fault coverage. This example is the first workflow acceptance
baseline, not proof of every format, language, integration or translation quality.
Implement it alongside the selected vertical slice; the document does not imply
that currently missing APIs or unresolved migration policies already exist.

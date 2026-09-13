---
name: tradusco-run
description: Drive a complete Tradusco cycle in a host repository through preparation, glossary and context decisions, translation, audit, resume and delivery.
---

# Tradusco run

Operate from the host repository. Resolve the integration config, normally
`.tradusco/config.json`, and read `traduscoRoot`. Invoke every Tradusco command
through that root. Never print the environment file or API keys.

1. Read the config and requested locale, source-key and regeneration scope.
2. Run `tools/run.js` with `--skip-translate --skip-delivery`. This extracts and
   syncs sources, refreshes deterministic glossary/context data and audits the
   prepared state without a model call or host delivery.
3. Run the glossary `report` and `next` commands. When candidates exist, read
   [the glossary procedure](../tradusco-glossary/SKILL.md) and submit decisions
   within the user's granted authority. Defer unresolved decisions instead of
   guessing.
4. Run the context `report` and `next` commands. When candidates exist, read
   [the context procedure](../tradusco-context/SKILL.md) and submit supported
   formulations. A `--needs-glossary` answer returns work to step 3. Continue
   until both queues are stable or the remaining optional work is deferred.
5. Run `tools/run.js --dry-run` with the exact requested selection. Treat its
   output as command and current-status inspection; it does not show complete
   model envelopes.
6. If the request authorizes translation, run the same scope without
   `--dry-run`. A normal run repeats deterministic preparation before model work,
   then audits and delivers. Do not broaden `--regenerate` or bypass
   `protectLangs`.
7. For a run expected to exceed one minute, use a PTY and verify the process exit
   and removal of `<projectDir>/.run.lock`. On partial failure, inspect the audit
   and locale `failures.jsonl`; retry the same bounded scope only when persisted
   progress makes the retry useful.

If the request covers preparation only, stop after step 5. Report the selected
scope, accepted/rejected/deferred decisions, translation outcome, unresolved
failures and delivery result that actually occurred.

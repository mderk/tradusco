---
name: tradusco-glossary
description: Prepare generated glossary terms and resolve queued terminology decisions before translation.
---

# Tradusco glossary

Resolve the integration config (normally `.tradusco/config.json`), read its
`traduscoRoot`, and invoke `tools/glossary.js` from that directory. Do not assume
the host repository contains `tools/`. Do not edit generated `terms` by hand.

1. Run `node <traduscoRoot>/tools/glossary.js report --config <config>`.
2. If configured source data changed, run `prepare` first without `--write`,
   inspect its counts, then repeat with `--write` when the preview is expected.
3. Run `next`. Use only its source examples and reviewed-language values as
   evidence. If product meaning or canon is unclear, ask the project owner.
4. Submit one decision with `--term` and either `--reject`, `--defer`, or
   `--mode`, `--note` and one `--translation LANG=VALUE` per reviewed language.
   Defer when evidence or authority is missing. Use `--json` only for batch input.
5. Run `lint`. Lint findings are review candidates;
   they do not authorize automatic translation edits or paid retries.

Accepted, rejected and deferred decisions persist, so do not reopen them without
new evidence or an explicit request. Use `reopen --term <term>` when that happens.

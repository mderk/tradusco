---
name: tradusco-glossary
description: Prepare generated glossary terms and resolve queued terminology decisions before translation.
---

# Tradusco glossary

Use the repository CLI; do not edit the generated `terms` section by hand.

1. Run `node tools/glossary.js report --config <path>`.
2. If configured source data changed, run `prepare` first without `--write`,
   inspect its counts, then repeat with `--write` when the preview is expected.
3. Run `next`. Use only its source examples and reviewed-language values as
   evidence. If product meaning or canon is unclear, ask the project owner.
4. Submit one decision with `--term` and either `--reject`, or `--mode`, `--note`
   and one `--translation LANG=VALUE` per reviewed language. Use `--json` only
   for batch input.
5. Run `lint`. Lint findings are review candidates;
   they do not authorize automatic translation edits or paid retries.

Accepted, rejected and deferred decisions persist, so do not reopen them without
new evidence or an explicit request.

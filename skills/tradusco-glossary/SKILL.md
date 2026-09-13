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
4. Write one JSON answer with the reported shape. Use `not_a_term` for a rejected
   candidate; otherwise provide `mode`, a short English `note`, and `t` for every
   reviewed language.
5. Run `submit --json <answer>`, then `lint`. Lint findings are review candidates;
   they do not authorize automatic translation edits or paid retries.

Accepted, rejected and deferred decisions persist, so do not reopen them without
new evidence or an explicit request.

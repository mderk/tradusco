---
name: tradusco-context
description: Preview, apply and complete translation context using a project's deterministic provider.
---

# Tradusco context

1. Run `node tools/context.js report --config <path>` and `apply` without
   `--write`.
2. Inspect the preview examples and revision. Existing generated rules run
   automatically; editing provider rules is separate work.
3. Apply the exact preview with `apply --write --expect <revision>`.
4. Run `next`. Read the reported project files and all references for each
   string. Do not infer product facts that the evidence does not establish.
5. Submit one short English context with `--group`, `--source` and `--context`.
   Use `--needs-glossary` instead of `--context` for a stable product term. Use
   `--json` only for batch input.
6. Repeat report/apply. Missing optional context remains visible but does not
   block translation.

Manual context is authoritative. Generated context fills empty cells and does
not refresh already filled generated values in this upgrade.

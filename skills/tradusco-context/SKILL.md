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
5. Submit one short English context per resolved string. Put stable product terms
   in `needs_glossary` with the context evidence instead of inventing a row note.
6. Repeat report/apply. Missing optional context remains visible but does not
   block translation.

Manual context is authoritative. Generated context fills empty cells and does
not refresh already filled generated values in this upgrade.

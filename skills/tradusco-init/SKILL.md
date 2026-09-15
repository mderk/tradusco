---
name: tradusco-init
description: Connect Tradusco to a new host project by inspecting its catalogs, scaffolding .tradusco state and implementing project-specific extraction, glossary, context and delivery adapters.
---

# Tradusco init

Work in the **host repository**. Read its agent instructions and
[`INTEGRATION_GUIDE.md`](../../INTEGRATION_GUIDE.md) before editing. Tradusco
uses exact base-language text as its current key; do not promise separate
translations for duplicate source text.

1. Find the host's source catalogs, target locales, existing translations,
   canonical terminology, context-bearing data and build command. Reuse existing
   readers and writers. Record which files the host owns and which values are
   known editorial; do not classify unknown existing translations by guesswork.
2. Create or reuse an interchange CSV with base, target-locale and `context`
   columns. Preserve existing host translations during extraction. Run
   [`scripts/scaffold.py`](scripts/scaffold.py) after the CSV exists to create
   `.tradusco/config.json` without overwriting an existing one.

   ```bash
   python3 <traduscoRoot>/skills/tradusco-init/scripts/scaffold.py \
     --host . --csv locale_src/translations.csv --base en --locales fr,de
   ```

   The script creates only the state directory and minimal config; the next
   step makes the host adapters real.
3. Implement only the adapters this project can support: `extractCommands`
   produce the CSV; `glossarySourceCommand` accepts `--output PATH` and writes
   generated `terms`; `contextProviderFile` exports `createApi(projectRoot)` with
   a non-empty revision plus relevant `csv`, `json`, `po` or `rules`; and
   `deliveryCommands` apply ready CSV values to the actual host catalogs and
   build them. The config lives in `.tradusco/`, so command cwd and
   `projectRoot` are that directory. Use `../` for host paths. If a project has
   no real source for glossary or context, omit that provider rather than invent
   terms or context. Set `artifactKeysCommand` only when the final build can
   expose exact source keys.
4. Set `protectLangs` to reviewed locales that must not be regenerated. Existing
   host values absent from Tradusco progress need an explicit import decision:
   bootstrap known machine progress or preview guarded back-sync for known
   editorial values. Do not run paid translation or delivery over unresolved
   existing values.
5. Verify each configured adapter alone, then run `node <traduscoRoot>/tools/run.js
   --config .tradusco/config.json --skip-translate --skip-delivery`. Confirm
   extraction/sync, glossary and context coverage, and that no real catalogs
   changed. `--dry-run` only lists commands and current status; it is not this
   preparation check. Then hand off to
   [`tradusco-run`](../tradusco-run/SKILL.md) for candidate decisions and
   translation when requested.

Keep all Tradusco state under `.tradusco/`. Project adapters and source catalogs
stay in the host's normal code/data locations. Report working adapters and any
remaining manual import or provider gaps.

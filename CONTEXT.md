# Translation context workflow

Tradusco stores per-row context in the configured `context` CSV column. The
workflow resolves it in this order: reviewed manual value, project column rule,
project domain rule, then general project rule. Generated values fill gaps;
manual values may replace an existing generated value.

## Project provider

`.tradusco/config.json` names the project directory and provider:

```json
{
  "projectDir": "shop",
  "contextProviderFile": "../scripts/context-provider.js"
}
```

The CommonJS provider exports `createApi(projectRoot)` plus any of `csv`, `json`,
`po` and `rules`. `createApi` returns a non-empty `revision` and the project data
methods used by its declarations:

- `table(file)` returns rows used by `csv` column callbacks;
- `json(file)` returns data visited by `json` key callbacks;
- `domainOf(source)` selects a `po` callback;
- `hit(source)` returns project evidence, including optional `refs`.

Column callbacks receive `(row, api)`, domain callbacks receive
`(source, api, hit)`, and rule callbacks receive `(match, api)`. Each returns one
context string or `null`. Product paths, data interpretation and wording remain
in this provider.

## Commands

```bash
node tools/context.js report --config .tradusco/config.json
node tools/context.js apply --config .tradusco/config.json
node tools/context.js apply --write --expect <preview-revision> --config .tradusco/config.json
node tools/context.js next --config .tradusco/config.json
node tools/context.js submit --group src/ui.js --source Mystery --context "Label for an unknown reward." --config .tradusco/config.json
node tools/context.js submit --group src/ui.js --source Weapon --needs-glossary "Equipment category." --config .tradusco/config.json
```

Accepted manual context defaults to `<projectDir>/contexts.json`.
For batch input, `submit --json <answer-file>` accepts the answer shape printed
by `next`.

`apply` is read-only by default and shows old/new values, resolution sources and
a revision derived from the source CSV, manual contexts, provider code and the
provider's data revision. A write with a stale revision is rejected.

`next` groups unresolved strings by project reference. `submit` accepts either a
manual context or `needs_glossary`; the latter enters the shared terminology
queue. Existing glossary terms are not requested again as row context.

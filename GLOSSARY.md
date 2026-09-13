# Translation glossary

Tradusco reads an optional `<project>/glossary.json`. If the file is absent,
translation continues without a glossary.

## Workflow tool

The first-upgrade workflow keeps generated terms and reviewed decisions in one
file while preserving their separate sections. Its project configuration is:

```json
{
  "projectDir": "shop",
  "glossarySourceCommand": ["node", "../scripts/build-glossary.js"]
}
```

The source command receives `--output <temporary-file>` and writes the generated
`terms` object. Preview is the default; `--write` replaces only `terms`:

```bash
node tools/glossary.js prepare --config .tradusco/config.json
node tools/glossary.js prepare --write --config .tradusco/config.json
node tools/glossary.js report --config .tradusco/config.json
node tools/glossary.js next --config .tradusco/config.json
node tools/glossary.js submit --json .tradusco/glossary-answer.json --config .tradusco/config.json
node tools/glossary.js lint --config .tradusco/config.json
```

By default the glossary and candidate decisions persist in
`<projectDir>/glossary.json`, `<projectDir>/not_terms.json` and
`<projectDir>/terms_queue.json`. The paths can be overridden with `glossaryFile`,
`glossaryRejectedFile` and `glossaryQueueFile`. The prompt and lint paths use the
same source matcher from `lib/glossary.py`. Agent operation is documented in
[`skills/tradusco-glossary/SKILL.md`](skills/tradusco-glossary/SKILL.md).

## File format

```json
{
  "_": {"modes": {}},
  "terms": {
    "Booty Chest": {
      "mode": "stem",
      "t": {"pl": "...", "ko": "..."},
      "scope": "ui"
    }
  },
  "manual": {}
}
```

`terms` is generated data. `manual` contains reviewed overrides. When the same
term exists in both sections, the complete `manual` entry replaces the generated
entry; fields are not merged individually.

## Matching modes

- `exact` matches only when the complete source phrase equals the glossary key.
- `stem` matches the key at a word boundary inside a phrase and permits a word
  suffix, such as plural or inflection.
- `keep` matches like `stem`, but tells the model to preserve the source term.
- `skip` never matches and is not sent to the model.

Optional matching fields:

- `cs: true` makes the term comparison case-sensitive.
- `near` is a regular expression that must also match the same source phrase.
  Invalid regular expressions make the entry inapplicable.
- `note` is author guidance passed to the model with a selected entry.
- `except` names a source catalog or catalogs where the entry must not apply. It
  is passed to the model, but can only be acted on when phrase context explicitly
  identifies that catalog.

A language value in `t` may be either a canonical string or a list of accepted
forms. In a list, the first form is canonical and later forms are permitted
inflections or variants. The prompt tells the model to prefer the canonical form
unless the phrase grammar requires another listed form.

Tradusco applies these rules locally. Selected entries also include `mode`, `cs`,
and `near` in the model request because the glossary is shared by the batch and
the model must know which individual phrases each entry applies to.

## Request selection

Only entries matching at least one phrase in the batch are considered. Their `t`
map is restricted to the target and reviewed reference languages of the request.
Reference-language forms clarify meaning when a target does not yet have a
reviewed glossary value; they are not target output. At most 20 entries are sent;
terms matching more batch phrases take priority. A typical request contains:

```json
{
  "glossary": [
    {
      "term": "Chaos",
      "mode": "stem",
      "cs": true,
      "near": "\\bFaction\\b",
      "t": {"pl": "...", "ko": "..."}
    }
  ],
  "phrases": [{"phrase": "Chaos Faction"}]
}
```

## Scope

`scope` describes where a terminology rule applies, for example `ui`, `dialogue`,
or `all`. It is not equivalent to gettext `msgctxt`: it filters a glossary rule
but does not make scope part of a source phrase's identity.

Current translation CSV files do not provide a deterministic scope for each row,
so Tradusco treats glossary scope as non-restrictive and does not send it to the
model. `except` is preserved as author metadata, but without an explicit catalog
in phrase context it cannot be applied reliably. Deterministic scope and catalog
filtering require an explicit row field supplied by the source project.

If identical source phrases need different translations in different contexts,
that requires a separate message-identity/storage change. Current `progress.json`
files are keyed by source phrase, so glossary scope alone cannot represent it.

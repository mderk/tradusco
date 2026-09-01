# Translation glossary

Tradusco reads an optional `<project>/glossary.json`. If the file is absent,
translation continues without a glossary.

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

Tradusco applies these rules locally. Selected entries also include `mode`, `cs`,
and `near` in the model request because the glossary is shared by the batch and
the model must know which individual phrases each entry applies to.

## Request selection

Only entries matching at least one phrase in the batch are considered. Their `t`
map is restricted to the target languages of the request. At most 20 entries are
sent; terms matching more batch phrases take priority. A typical request contains:

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
model. Adding scope filtering requires an explicit row field supplied by the
source project.

If identical source phrases need different translations in different contexts,
that requires a separate message-identity/storage change. Current `progress.json`
files are keyed by source phrase, so glossary scope alone cannot represent it.

# Deferred tooling ideas

> **Status: backlog.** These items do not block the current standalone workflow.
> Promote an item only when a real integration or a reproducible failure requires it.

| Idea | Revisit when |
| --- | --- |
| Stable-ID storage, owner namespaces and legacy identity migration | A target project cannot use the current source-as-key convention. |
| Domain-separated translations for the same source text | A host delivery format can preserve separate domain values. |
| General transactions and concurrent-writer coordination | Supported workflows actually write the same project concurrently or lose accepted edits at an uncovered boundary. |
| Cache instrumentation and glossary-based batch ordering | Provider usage data shows a material avoidable cache cost. |
| Output-volume batch sizing | A bounded, reproducible run shows that current split-on-failure is inadequate. |
| Additional structural, semantic and language-specific checks, style baselines and quoted-UI references | Existing project checks have measured precision and a second project needs them. |
| Multiple-adapter parity suite | A second real catalogue adapter is added. |
| Reference/example translation-quality experiments | A separate experiment has a fixed dataset, model and spending budget. |
| Model bakeoff utility | A model comparison is requested independently of the ordinary run. |
| Whitespace rekey migration | A real project needs an explicit old-key to new-key mapping. |
| Canon-gap reporting and canon import | A project needs a repeatable canon-gap workflow. |
| Migration of legacy override layers | A target project must consolidate existing competing override files. |
| Inferring editorial origin for edits made outside supported tools | A project cannot route editorial changes through guarded edits or back-sync. |
| Refreshing already stored context after a rule changes | The transferred context workflow must update generated values rather than only fill gaps. |
| QA and review agent skills | Repeated agent operation shows that the run skill and direct review CLI are insufficient. |
| Long-run heartbeat and detached supervision | A real run must survive loss of its invoking agent session. |

# Tooling ideas deferred after the first upgrade

> **Status: backlog.** These items do not block the G1–G11 transfer in
> `REWORK_PLAN_TOOLING.md`. Promote an item only when a real integration or a
> reproducible failure requires it.

| Idea | Revisit when |
| --- | --- |
| Stable-ID storage, owner namespaces and legacy identity migration | A target project cannot use the current source-as-key convention. |
| Domain-separated translations for the same source text | A host delivery format can preserve separate domain values. |
| General transactions and concurrent-writer coordination | Supported workflows actually write the same project concurrently or lose accepted edits at an uncovered boundary. |
| Cache instrumentation and glossary-based batch ordering | Provider usage data shows a material avoidable cache cost. |
| Output-volume batch sizing | A bounded, reproducible run shows that current split-on-failure is inadequate. |
| New semantic and language-specific checks, style baselines and quoted-UI references | Existing project checks have measured precision and a second project needs them. |
| Automated `tradusco-init` interview and scaffolding | Repeated manual setup of another project proves the setup contract stable. |
| Multiple-adapter parity suite | A second real catalogue adapter is added. |
| Reference/example translation-quality experiments | A separate experiment has a fixed dataset, model and spending budget. |

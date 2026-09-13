from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable


@dataclass(frozen=True, eq=False)
class GlossaryRule:
    term: str
    data: dict[str, Any]
    order: int

    @cached_property
    def pattern(self) -> re.Pattern[str]:
        flags = 0 if self.data.get("cs") else re.IGNORECASE
        return re.compile(rf"(?<!\w){re.escape(self.term)}\w*", flags)

    def spans(self, phrase: str) -> list[tuple[int, int]]:
        mode = self.data.get("mode", "exact")
        if mode == "skip":
            return []
        if mode == "exact":
            matches = (
                phrase == self.term
                if self.data.get("cs")
                else phrase.casefold() == self.term.casefold()
            )
            spans = [(0, len(phrase))] if matches else []
        else:
            spans = [match.span() for match in self.pattern.finditer(phrase)]
        near = self.data.get("near")
        if spans and isinstance(near, str) and near:
            try:
                if re.search(near, phrase) is None:
                    return []
            except re.error:
                return []
        return spans


def glossary_rules(glossary: dict[str, Any]) -> list[GlossaryRule]:
    merged: dict[str, dict[str, Any]] = {}
    for section in ("terms", "manual"):
        value = glossary.get(section)
        if isinstance(value, dict):
            merged.update(
                (str(term), data)
                for term, data in value.items()
                if term and isinstance(data, dict)
            )
    return [
        GlossaryRule(term, data, order)
        for order, (term, data) in enumerate(merged.items())
    ]


_SHORT_WORDS = re.compile(
    r"^(с|в|на|и|of|the|de|du|la|le|and|à|von|der|die|das)$", re.I
)


def _stem(word: str) -> str:
    if len(word) <= 2:
        return word
    if len(word) <= 5:
        return word[:-1]
    return word[: (len(word) * 3 + 4) // 5]


def translation_matches(value: str, form: str, *, stem: bool) -> bool:
    haystack = value.casefold()
    words = [word for word in form.split() if word and not _SHORT_WORDS.match(word)]
    if not words:
        return form.casefold() in haystack
    return all((_stem(word) if stem else word).casefold() in haystack for word in words)


def lint_glossary(
    glossary: dict[str, Any],
    rows: Iterable[dict[str, str]],
    base_language: str,
    languages: Iterable[str],
) -> list[dict[str, str]]:
    rules = glossary_rules(glossary)
    own_terms = {rule.term.casefold() for rule in rules}
    findings: list[dict[str, str]] = []
    for row in rows:
        source = str(row.get(base_language, ""))
        for rule in rules:
            mode = str(rule.data.get("mode", "exact"))
            if mode in {"skip", "keep"} or not rule.spans(source):
                continue
            if mode != "exact" and source.strip().casefold() in own_terms:
                continue
            values = rule.data.get("t")
            if not isinstance(values, dict):
                continue
            for language in languages:
                translation = row.get(language, "")
                forms = values.get(language)
                forms = [forms] if isinstance(forms, str) else forms
                if not translation or not isinstance(forms, list) or not forms:
                    continue
                valid_forms = [form for form in forms if isinstance(form, str) and form]
                if valid_forms and not any(
                    translation_matches(translation, form, stem=mode == "stem")
                    for form in valid_forms
                ):
                    findings.append(
                        {
                            "language": language,
                            "term": rule.term,
                            "source": source,
                            "translation": translation,
                            "expected": valid_forms[0],
                        }
                    )
    return findings


def glossary_coverage(
    glossary: dict[str, Any], sources: Iterable[str]
) -> list[dict[str, int | str]]:
    phrases = list(sources)
    return [
        {
            "term": rule.term,
            "matched_rows": sum(bool(rule.spans(phrase)) for phrase in phrases),
            "whole_phrase_rows": sum(
                bool(rule.spans(phrase)) and phrase.casefold() == rule.term.casefold()
                for phrase in phrases
            ),
        }
        for rule in glossary_rules(glossary)
    ]

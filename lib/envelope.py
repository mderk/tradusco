from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from lib.utils import is_valid_translation, placeholders_match


class Example(BaseModel):
    phrase: str
    t: dict[str, str]


class PhraseEnvelope(BaseModel):
    phrase: str
    context: str | None = None
    reference: dict[str, str] | None = None
    examples: list[Example] | None = None


class BatchEnvelope(BaseModel):
    glossary: list[dict[str, Any]] | None = None
    phrases: list[PhraseEnvelope]


@dataclass(frozen=True, eq=False)
class _Rule:
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

        case_sensitive = bool(self.data.get("cs"))
        if mode == "exact":
            matches = phrase == self.term if case_sensitive else phrase.casefold() == self.term.casefold()
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


class EnvelopeBuilder:
    def __init__(
        self,
        glossary: dict[str, Any],
        translations: list[dict[str, str]],
        base_language: str,
        dst_languages: list[str],
        reference_languages: list[str],
    ) -> None:
        merged = {
            **self._section(glossary.get("terms")),
            **self._section(glossary.get("manual")),
        }
        self.rules = [
            _Rule(term, data, order)
            for order, (term, data) in enumerate(merged.items())
            if term and isinstance(data, dict)
        ]
        self.translations = translations
        self.base_language = base_language
        self.dst_languages = dst_languages
        self.reference_languages = reference_languages
        self.glossary_languages = list(
            dict.fromkeys([*dst_languages, *reference_languages])
        )
        self.matches = [
            self._matches(row.get(base_language, "")) for row in translations
        ]
        self.neighbors: dict[str, list[int]] = defaultdict(list)
        for index, row_matches in enumerate(self.matches):
            if row_matches:
                phrase = translations[index].get(base_language, "")
                self.neighbors[self._skeleton(phrase, row_matches)].append(index)

    @staticmethod
    def _section(value: object) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        return {
            str(term): data
            for term, data in value.items()
            if isinstance(data, dict)
        }

    def _matches(self, phrase: str) -> list[tuple[_Rule, list[tuple[int, int]]]]:
        return [
            (rule, spans)
            for rule in self.rules
            if (spans := rule.spans(phrase))
        ]

    @staticmethod
    def _skeleton(
        phrase: str, matches: list[tuple[_Rule, list[tuple[int, int]]]]
    ) -> str:
        candidates = sorted(
            (span for _rule, spans in matches for span in spans),
            key=lambda span: (span[0], -(span[1] - span[0])),
        )
        selected: list[tuple[int, int]] = []
        for span in candidates:
            if not selected or span[0] >= selected[-1][1]:
                selected.append(span)
        for start, end in reversed(selected):
            phrase = f"{phrase[:start]}{{term}}{phrase[end:]}"
        return phrase

    def _prompt_glossary(
        self, batch_matches: list[list[tuple[_Rule, list[tuple[int, int]]]]]
    ) -> list[dict[str, Any]]:
        counts = Counter(rule for matches in batch_matches for rule, _spans in matches)
        selected: list[dict[str, Any]] = []
        for rule, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].order)
        ):
            translations = rule.data.get("t")
            filtered = (
                {
                    language: translations[language]
                    for language in self.glossary_languages
                    if isinstance(translations, dict)
                    and language in translations
                    and (
                        isinstance(translations[language], str)
                        or (
                            isinstance(translations[language], list)
                            and all(
                                isinstance(form, str)
                                for form in translations[language]
                            )
                        )
                    )
                }
                if isinstance(translations, dict)
                else {}
            )
            mode = str(rule.data.get("mode", "exact"))
            if not filtered and mode != "keep":
                continue
            entry: dict[str, Any] = {"term": rule.term, "mode": mode}
            if rule.data.get("cs"):
                entry["cs"] = True
            near = rule.data.get("near")
            if isinstance(near, str) and near:
                entry["near"] = near
            note = rule.data.get("note")
            if isinstance(note, str) and note:
                entry["note"] = note
            excluded = rule.data.get("except")
            if isinstance(excluded, (str, list)) and excluded:
                entry["except"] = excluded
            if filtered:
                entry["t"] = filtered
            selected.append(entry)
            if len(selected) == 20:
                break
        return selected

    def _examples(
        self,
        row_index: int,
        phrase: str,
        matches: list[tuple[_Rule, list[tuple[int, int]]]],
    ) -> list[Example]:
        if not matches:
            return []
        candidates = self.neighbors.get(self._skeleton(phrase, matches), [])
        candidates = sorted(
            (index for index in candidates if index != row_index),
            key=lambda index: abs(
                len(self.translations[index].get(self.base_language, "")) - len(phrase)
            ),
        )
        examples: list[Example] = []
        for index in candidates:
            row = self.translations[index]
            source = row.get(self.base_language, "")
            translated = {
                language: value
                for language in self.dst_languages
                if isinstance((value := row.get(language)), str)
                and is_valid_translation(value)
                and placeholders_match(source, value)[0]
            }
            if translated:
                examples.append(Example(phrase=source, t=translated))
            if len(examples) == 3:
                break
        return examples

    def build(
        self,
        phrases: list[tuple[str, str | None]],
        phrase_indices: dict[str, int],
    ) -> BatchEnvelope:
        batch_matches = [self.matches[phrase_indices[phrase]] for phrase, _ in phrases]
        envelopes: list[PhraseEnvelope] = []
        for (phrase, context), matches in zip(phrases, batch_matches):
            row_index = phrase_indices[phrase]
            row = self.translations[row_index]
            reference = {
                language: value
                for language in self.reference_languages
                if isinstance((value := row.get(language)), str) and value.strip()
            }
            examples = self._examples(row_index, phrase, matches)
            envelopes.append(
                PhraseEnvelope(
                    phrase=phrase,
                    context=context or None,
                    reference=reference or None,
                    examples=examples or None,
                )
            )
        glossary = self._prompt_glossary(batch_matches)
        return BatchEnvelope(glossary=glossary or None, phrases=envelopes)

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LanguageTranslations(BaseModel):
    """Translations for one requested language."""

    model_config = ConfigDict(extra="forbid")

    language: str
    translations: list[str]


class TranslationsResponse(BaseModel):
    """Structured Outputs contract for a translation batch."""

    model_config = ConfigDict(extra="forbid")

    translations: list[LanguageTranslations] = Field(
        ...,
        description=(
            "One block per requested language. Each block contains translations "
            "in the same order as input phrases."
        ),
    )


__all__ = ["LanguageTranslations", "TranslationsResponse"]

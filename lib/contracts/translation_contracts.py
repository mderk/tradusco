from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FailureCategory = Literal[
    "refusal",
    "content_filter",
    "parse_error",
    "placeholder_mismatch",
    "invalid_artifact_rejected",
    "other",
]


class TranslationFailure(BaseModel):
    """An individual phrase-level failure reported by the model."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(
        ...,
        description="0-based index of the phrase in the input list.",
    )
    phrase: str = Field(
        ...,
        description="The original source phrase (base language).",
    )
    category: FailureCategory = Field(
        ...,
        description="Failure category, used for reporting and fallback routing.",
    )
    message: str | None = Field(
        None,
        description="Optional short explanation (may be empty).",
    )


class TranslationsResponse(BaseModel):
    """Structured Outputs contract for a translation batch."""

    model_config = ConfigDict(extra="forbid")

    translations: list[str] = Field(
        ...,
        description=(
            "Array of translations from source language to target language "
            "in the same order as input phrases."
        ),
    )

    failures: list[TranslationFailure] = Field(
        default_factory=list,
        description=(
            "Optional list of failures for phrases that could not be translated. "
            "When present, callers can use this to route retries/fallbacks."
        ),
    )


__all__ = ["FailureCategory", "TranslationFailure", "TranslationsResponse"]


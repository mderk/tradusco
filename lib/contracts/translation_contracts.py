from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


__all__ = ["TranslationsResponse"]


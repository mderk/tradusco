from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

FailureCategory = Literal[
    "refusal",
    "network_error",
    "parse_error",
    "placeholder_mismatch",
    "invalid_artifact_rejected",
    "other",
]


def batch_error_kind_to_category(kind: str) -> FailureCategory:
    if kind == "blocked":
        return "refusal"
    if kind in {"rate_limit", "auth_error"}:
        return "network_error"
    if kind == "model_error":
        return "parse_error"
    return "other"


def make_failure_record(
    *,
    model: str,
    phrase: str,
    category: FailureCategory,
    message: str,
    method: str | None = None,
) -> dict[str, str | None]:
    return {
        "ts": datetime.now(UTC).isoformat(),
        "model": model,
        "method": method,
        "phrase": phrase,
        "category": category,
        "message": message[:500] if message else "",
    }


__all__ = [
    "FailureCategory",
    "batch_error_kind_to_category",
    "make_failure_record",
]

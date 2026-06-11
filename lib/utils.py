"""
Utility functions and classes for the translation project.
"""

import re

from pydantic import BaseModel


class Config(BaseModel):
    """Project configuration model"""

    name: str
    sourceFile: str
    languages: list[str]
    baseLanguage: str
    keyColumn: str


# ---------------------------------------------------------------------------
# Placeholder / Lingui-tag extraction
#
# Shared by the live translation path (TranslationTool) and the offline audit
# script so both enforce the exact same rules.
# ---------------------------------------------------------------------------

_CURLY_TOKEN_RE = re.compile(r"\{[^}]+\}")
# Lingui uses numeric tags like <0>...</0>. Keep them intact.
_LINGUI_TAG_RE = re.compile(r"</?\d+/?\s*>")
# Detects a string that *is* (the start of) our expected JSON output shape,
# e.g. `{"translations": [...]`. Used to reject model scaffolding that leaks
# through parsing. Deliberately anchored so legitimate text merely *containing*
# the word "translations" is not rejected.
_TRANSLATIONS_KEY_RE = re.compile(r"""^\{?\s*["']translations["']\s*:""")


def extract_curly_tokens(text: str) -> set[str]:
    return set(_CURLY_TOKEN_RE.findall(text or ""))


def extract_lingui_tags(text: str) -> set[str]:
    return set(_LINGUI_TAG_RE.findall(text or ""))


def placeholders_match(source: str, translation: str) -> tuple[bool, str]:
    """
    Ensure a translation preserves placeholders (`{num}`) and Lingui tags
    (`<0>...</0>`) so runtime interpolation is not broken.
    """
    src_tokens = extract_curly_tokens(source)
    dst_tokens = extract_curly_tokens(translation)
    if src_tokens != dst_tokens:
        return (
            False,
            f"curly placeholders mismatch: src={sorted(src_tokens)} dst={sorted(dst_tokens)}",
        )

    src_tags = extract_lingui_tags(source)
    dst_tags = extract_lingui_tags(translation)
    if src_tags != dst_tags:
        return (
            False,
            f"lingui tags mismatch: src={sorted(src_tags)} dst={sorted(dst_tags)}",
        )

    return True, ""


def looks_like_json_artifact(value: object) -> bool:
    """
    True when the whole value looks like JSON scaffolding from our expected
    output shape rather than an actual translation.

    Empty strings are NOT artifacts (they are "missing"); callers that treat
    empties as invalid should check emptiness separately or use
    ``is_valid_translation``.
    """
    s = str(value or "").strip()
    if not s:
        return False
    if s in {"{", "}", "[", "]", "{}", "[]"}:
        return True
    # The whole string is (the start of) our `{"translations": [...]}` object
    # or a dangling `"translations":` key line. Anchored, so it does not match
    # legitimate text that merely contains the word "translations".
    if _TRANSLATIONS_KEY_RE.match(s):
        return True
    return False


def is_valid_translation(value: object) -> bool:
    """A value is a usable translation if it is non-empty and not scaffolding."""
    s = str(value or "").strip()
    if not s:
        return False
    return not looks_like_json_artifact(s)


def validate_translation_text(translation: object) -> tuple[bool, str]:
    """
    Reject obvious model-output artifacts that can slip through parsing.

    Returns ``(ok, reason)`` so callers can log why a value was rejected.
    """
    s = str(translation or "").strip()
    if not s:
        return False, "empty translation"
    if looks_like_json_artifact(s):
        return False, "looks like JSON scaffolding"
    return True, ""

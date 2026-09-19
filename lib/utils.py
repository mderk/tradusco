"""
Utility functions and classes for the translation project.
"""

import math
import re
import unicodedata

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

# Lingui uses numeric tags like <0>...</0>. Keep them intact.
_LINGUI_TAG_RE = re.compile(r"</?\d+/?\s*>")
# ICU MessageFormat arguments whose body is a set of `key {message}` branches.
# Their branches are language-specific (Russian needs `few`/`many`, English
# does not), so they are compared structurally rather than as literal text.
_ICU_BRANCHED_TYPES = frozenset({"plural", "selectordinal", "select"})
_PLURAL_CATEGORIES = frozenset({"zero", "one", "two", "few", "many", "other"})
_ICU_ARG_HEAD_RE = re.compile(
    r"^\s*([^,\s]+)\s*(?:,\s*([A-Za-z]+)\s*(?:,(.*))?)?$", re.DOTALL
)
_ICU_BRANCH_KEY_RE = re.compile(
    r"\s*(?:offset\s*:\s*\d+\s*)?(=\d+|[A-Za-z_][\w-]*)\s*\{"
)


class IcuArg:
    """One `{...}` argument of a message, parsed just far enough to compare."""

    __slots__ = ("raw", "name", "kind", "branches")

    def __init__(
        self, raw: str, name: str = "", kind: str = "", branches: dict | None = None
    ):
        # The argument text including braces.
        self.raw = raw
        self.name = name
        # "plural" / "select" / "selectordinal"; "" for anything else.
        self.kind = kind
        # Branch key -> branch message (branched kinds only).
        self.branches = branches or {}

    @property
    def token(self) -> str:
        """What the argument must look like in a translation."""
        return f"{{{self.name}, {self.kind}}}" if self.kind else self.raw


def _matching_brace(text: str, start: int) -> int:
    """Index of the `}` closing the `{` at ``start``, or -1 when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _parse_branches(body: str) -> dict[str, str] | None:
    """`one {# file} other {# files}` -> {"one": "# file", "other": "# files"}."""
    branches: dict[str, str] = {}
    pos = 0
    while pos < len(body):
        if body[pos:].strip() == "":
            break
        m = _ICU_BRANCH_KEY_RE.match(body, pos)
        if not m:
            return None
        end = _matching_brace(body, m.end() - 1)
        if end < 0:
            return None
        branches[m.group(1)] = body[m.end() : end]
        pos = end + 1
    return branches


def _parse_arg(raw: str) -> IcuArg:
    m = _ICU_ARG_HEAD_RE.match(raw[1:-1])
    if not m:
        return IcuArg(raw)
    name, kind, body = m.group(1), (m.group(2) or "").lower(), m.group(3)
    if kind not in _ICU_BRANCHED_TYPES or body is None:
        return IcuArg(raw)
    branches = _parse_branches(body)
    if branches is None:
        return IcuArg(raw)
    return IcuArg(raw, name, kind, branches)


def parse_icu_args(text: str) -> list[IcuArg]:
    """
    Top-level `{...}` arguments of a message, in order. Plain `{name}` and
    `{n, number}` are kept verbatim; plural/select arguments get their branches.
    An unbalanced `{` is treated as literal text, like the old regex did.
    """
    args: list[IcuArg] = []
    text = text or ""
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        end = _matching_brace(text, start)
        if end < 0:
            break
        if end > start + 1:
            args.append(_parse_arg(text[start : end + 1]))
        pos = end + 1
    return args


def _flatten_icu(text: str) -> str:
    """Message with every branched argument replaced by its `other` branch."""
    out: list[str] = []
    text = text or ""
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        end = _matching_brace(text, start)
        if end < 0:
            break
        out.append(text[pos:start])
        arg = (
            _parse_arg(text[start : end + 1])
            if end > start + 1
            else IcuArg(text[start : end + 1])
        )
        if arg.kind:
            other = arg.branches.get("other") or next(iter(arg.branches.values()), "")
            out.append(_flatten_icu(other).replace("#", "0"))
        else:
            out.append(arg.raw)
        pos = end + 1
    out.append(text[pos:])
    return "".join(out)


def _nested_tokens(arg: IcuArg) -> set[str]:
    """Placeholders used anywhere inside a branched argument, `#` included."""
    tokens: set[str] = set()
    for body in arg.branches.values():
        if arg.kind != "select" and "#" in body:
            tokens.add("#")
        for inner in parse_icu_args(body):
            tokens.add(inner.token)
            tokens |= _nested_tokens(inner)
    return tokens


def _branched_arg_matches(src: IcuArg, dst: IcuArg) -> str:
    """Empty string when ``dst`` is a valid translation of ``src``, else why not."""
    where = f"{src.token}"
    if src.kind == "select":
        if set(src.branches) != set(dst.branches):
            return f"{where}: select keys differ: src={sorted(src.branches)} dst={sorted(dst.branches)}"
    else:
        if "other" not in dst.branches:
            return f"{where}: missing `other` branch"
        bad = [
            k
            for k in dst.branches
            if not (k.startswith("=") or k in _PLURAL_CATEGORIES)
        ]
        if bad:
            return f"{where}: unknown plural categories {sorted(bad)}"
        src_exact = {k for k in src.branches if k.startswith("=")}
        dst_exact = {k for k in dst.branches if k.startswith("=")}
        if src_exact != dst_exact:
            return f"{where}: exact-value branches differ: src={sorted(src_exact)} dst={sorted(dst_exact)}"
    src_inner, dst_inner = _nested_tokens(src), _nested_tokens(dst)
    if src_inner != dst_inner:
        return f"{where}: nested placeholders mismatch: src={sorted(src_inner)} dst={sorted(dst_inner)}"
    return ""


# Detects a string that *is* (the start of) our expected JSON output shape,
# e.g. `{"translations": [...]`. Used to reject model scaffolding that leaks
# through parsing. Deliberately anchored so legitimate text merely *containing*
# the word "translations" is not rejected.
_TRANSLATIONS_KEY_RE = re.compile(r"""^\{?\s*["']translations["']\s*:""")


def extract_curly_tokens(text: str) -> set[str]:
    """
    Placeholders a translation has to keep. Plain `{name}` tokens are returned
    verbatim; an ICU plural/select collapses to `{name, plural}` because its
    branches are allowed to differ between languages.
    """
    return {arg.token for arg in parse_icu_args(text)}


def extract_lingui_tags(text: str) -> set[str]:
    return set(_LINGUI_TAG_RE.findall(text or ""))


def placeholders_match(source: str, translation: str) -> tuple[bool, str]:
    """
    Ensure a translation preserves placeholders (`{num}`), ICU plural/select
    arguments and Lingui tags (`<0>...</0>`) so runtime interpolation is not
    broken.
    """
    src_args = parse_icu_args(source)
    dst_args = parse_icu_args(translation)
    src_tokens = {a.token for a in src_args}
    dst_tokens = {a.token for a in dst_args}
    if src_tokens != dst_tokens:
        return (
            False,
            f"curly placeholders mismatch: src={sorted(src_tokens)} dst={sorted(dst_tokens)}",
        )
    for src in src_args:
        if not src.kind:
            continue
        for dst in dst_args:
            if dst.token != src.token:
                continue
            reason = _branched_arg_matches(src, dst)
            if reason:
                return False, f"ICU argument mismatch: {reason}"

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


# ---------------------------------------------------------------------------
# Length control
#
# UI labels live in fixed-width buttons, tabs and badges: a translation that is
# much longer than the source overflows or gets clipped. Long prose is exempt —
# there the ratio says nothing useful — so only short source strings are judged.
# ---------------------------------------------------------------------------

# Anything that renders at runtime rather than as authored text: it costs the
# same width in every language, so it must not skew the comparison.
_MEASURE_STRIP_RE = re.compile(r"\{[^}]*\}|</?\d+/?\s*>|</?[a-zA-Z][^>]*>|%[sd]")

# A source that ends a sentence is a line of dialogue or a description, not a
# caption on a button — its length is free.
_SENTENCE_END_RE = re.compile(r"[.!?…。！？：:]$")
# `HP`, `EXP`, `Lv.`, `Atk.` — deliberately shortened by the designer because the
# widget is tiny, so the translation has to stay tiny as well. Ordinary short
# words (`Ash`, `Trap`) are not abbreviations and get the normal allowance.
_ABBREV_RE = re.compile(r"^(?:[A-Z]{2,4}|[A-Za-z]{2,4}\.)$")

LENGTH_DEFAULTS = {
    # Above this display width the string is prose, not a UI label.
    "maxSourceWidth": 40,
    # Above this many words it is a phrase, not a caption.
    "maxSourceWords": 6,
    # Allowed growth for a normal label.
    "maxRatio": 1.9,
    # Short labels need absolute headroom too: 1.9 * 4 is only 7 characters,
    # and a name transliterated into katakana costs more than that by itself.
    "minSlack": 6,
    # Absolute headroom for an abbreviation, on top of the source width.
    "abbrevSlack": 2,
    # Extra sources to treat as abbreviations beyond the automatic pattern.
    "abbrevSources": ["Lvl", "Atk", "Def", "Spd", "Crit"],
}


def display_width(text: str) -> int:
    """
    Width in terminal/UI cells: CJK and fullwidth characters take two, combining
    marks take none. Comparing widths (not character counts) is what makes a
    Japanese label comparable to an English one.
    """
    width = 0
    for ch in str(text or ""):
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def measurable_text(text: str) -> str:
    """Source text with placeholders and markup removed, whitespace collapsed."""
    # A plural/select renders as one of its branches, so measure the `other` one.
    stripped = _MEASURE_STRIP_RE.sub("", _flatten_icu(str(text or "")))
    return re.sub(r"\s+", " ", stripped).strip()


def is_abbreviation(source: str, options: dict | None = None) -> bool:
    """True when the source itself is a shortened form, not a full word."""
    opts = {**LENGTH_DEFAULTS, **(options or {})}
    s = measurable_text(source)
    if s in set(opts.get("abbrevSources") or []):
        return True
    return bool(_ABBREV_RE.match(s))


def length_limit(source: str, options: dict | None = None) -> int:
    """Maximum allowed display width of a translation for a given source."""
    opts = {**LENGTH_DEFAULTS, **(options or {})}
    source_width = display_width(measurable_text(source))
    if is_abbreviation(source, opts):
        return source_width + int(opts["abbrevSlack"])
    by_ratio = math.ceil(source_width * float(opts["maxRatio"]))
    return max(by_ratio, source_width + int(opts["minSlack"]))


def length_within_limit(
    source: str, translation: str, options: dict | None = None
) -> tuple[bool, str]:
    """
    Check that a short UI string did not grow beyond what its widget can show.

    Returns ``(ok, reason)``. Prose and empty strings are always ``ok``: the rule
    is deliberately limited to strings short enough to be labels.
    """
    opts = {**LENGTH_DEFAULTS, **(options or {})}
    src = measurable_text(source)
    dst = measurable_text(translation)
    if not src or not dst:
        return True, ""

    src_width = display_width(src)
    if src_width == 0 or src_width > int(opts["maxSourceWidth"]):
        return True, ""
    if len(src.split()) > int(opts["maxSourceWords"]):
        return True, ""
    if _SENTENCE_END_RE.search(src):
        return True, ""

    dst_width = display_width(dst)
    limit = length_limit(src, opts)
    if dst_width <= limit:
        return True, ""

    kind = "abbreviation" if is_abbreviation(src, opts) else "label"
    return (
        False,
        f"{kind} too long: width {dst_width} > {limit} "
        f"(source {src_width}, ratio {dst_width / src_width:.1f})",
    )


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

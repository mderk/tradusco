#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

# Mapping for single-pass unescape of PO escape sequences
_ESCAPE_MAP = {"\\\\": "\\", "\\n": "\n", "\\t": "\t", '\\"': '"'}

# Headers that change on every extract even without real content changes
_VOLATILE_HEADER_RE = re.compile(
    r'^"(POT-Creation-Date|PO-Revision-Date|Report-Msgid-Bugs-To):.*"$',
    re.MULTILINE,
)


def escape_po_string(s: str) -> str:
    """
    Escape a string for PO file format.

    Order matters: backslash must be escaped first.
    """
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s


def unescape_po_string(s: str) -> str:
    """
    Unescape a PO string.

    Uses single-pass left-to-right matching to correctly handle sequences like
    ``\\\\n`` (literal backslash + n) vs ``\\n`` (newline).
    """
    return re.sub(r'\\[nt"\\]', lambda m: _ESCAPE_MAP[m.group(0)], s)


def format_po_msgstr(translation: str) -> str:
    """
    Format a translation as a PO msgstr line (or multiline block).

    For strings containing newlines, produces proper multiline PO format:
        msgstr ""
        "first line\\n"
        "second line"
    """
    escaped = escape_po_string(translation)

    if "\\n" in escaped:
        segments = escaped.split("\\n")
        parts = ['msgstr ""']
        for i, segment in enumerate(segments):
            suffix = "\\n" if i < len(segments) - 1 else ""
            if segment or suffix:
                parts.append(f'"{segment}{suffix}"')
        return "\n".join(parts)

    return f'msgstr "{escaped}"'


def is_fuzzy(block: str) -> bool:
    """
    Check if a PO block has the #, fuzzy flag.

    Parses comma-separated flags to match 'fuzzy' exactly.
    """
    for line in block.split("\n"):
        if line.startswith("#,"):
            flags = [f.strip() for f in line[2:].split(",")]
            if "fuzzy" in flags:
                return True
    return False


def strip_fuzzy_flag(block: str) -> str:
    """
    Remove the fuzzy flag from a PO block's flags line.

    If 'fuzzy' is the only flag, removes the entire #, line.
    If there are other flags, removes only 'fuzzy'.
    """
    lines = block.split("\n")
    new_lines: list[str] = []
    for line in lines:
        if line.startswith("#,") and "fuzzy" in line:
            flags = [f.strip() for f in line[2:].split(",")]
            flags = [f for f in flags if f != "fuzzy"]
            if flags:
                new_lines.append("#, " + ", ".join(flags))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def strip_volatile_headers(content: str) -> str:
    """Remove headers that change every extract (dates, bug report url)."""
    return _VOLATILE_HEADER_RE.sub("", content)


def snapshot_files(paths: list[Path]) -> dict[Path, str]:
    """Read contents of files that exist, return {path: content}."""
    snap: dict[Path, str] = {}
    for p in paths:
        if p.exists():
            snap[p] = p.read_text(encoding="utf-8")
    return snap


def restore_date_only_changes(snapshot: dict[Path, str]) -> list[Path]:
    """
    Restore files where only volatile headers changed.

    Returns list of restored paths.
    """
    restored: list[Path] = []
    for path, old_content in snapshot.items():
        if not path.exists():
            continue
        new_content = path.read_text(encoding="utf-8")
        if new_content == old_content:
            continue
        if strip_volatile_headers(new_content) == strip_volatile_headers(old_content):
            path.write_text(old_content, encoding="utf-8")
            restored.append(path)
    return restored


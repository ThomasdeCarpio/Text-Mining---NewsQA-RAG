"""Deterministic parsing for generated answers and numbered citations."""

from __future__ import annotations

import re
from collections.abc import Sequence


_CITATION_GROUP_PATTERN = re.compile(
    r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]"
)
_ANSWER_LABEL_PATTERN = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*", re.IGNORECASE)
_REFERENCE_LINE_PATTERN = re.compile(
    r"^\s*(?:references?|sources?|citations?)\s*:\s*"
    r"(?:\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]\s*)+$",
    re.IGNORECASE,
)


def parse_citation_indices(text: str) -> list[int]:
    """Return unique numbered citations in their first-occurrence order."""

    indices: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_GROUP_PATTERN.finditer(text or ""):
        for value in re.findall(r"\d+", match.group(1)):
            index = int(value)
            if index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def split_citation_indices(
    text: str, context_count: int
) -> tuple[list[int], list[int]]:
    """Split parsed citations into valid and out-of-range indices."""

    indices = parse_citation_indices(text)
    valid = [index for index in indices if 1 <= index <= context_count]
    invalid = [index for index in indices if index not in valid]
    return valid, invalid


def cited_chunk_ids(text: str, ordered_chunk_ids: Sequence[str]) -> tuple[list[int], list[int], list[str]]:
    """Map citations in ``text`` to an ordered generation context."""

    valid, invalid = split_citation_indices(text, len(ordered_chunk_ids))
    return valid, invalid, [ordered_chunk_ids[index - 1] for index in valid]


def strip_citations(text: str) -> str:
    """Remove supported numbered citation syntax without changing answer content."""

    cleaned = _CITATION_GROUP_PATTERN.sub("", text or "")
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    return cleaned


def extract_answer_text(response: str) -> str:
    """Extract scoreable answer text from the supported response contracts.

    This removes citation markers, a leading ``Answer:`` label, and a standalone
    references line. It intentionally does not summarize, select facts, or
    otherwise rewrite the generated answer.
    """

    lines = [
        line
        for line in (response or "").replace("\r\n", "\n").split("\n")
        if not _REFERENCE_LINE_PATTERN.fullmatch(line)
    ]
    cleaned = "\n".join(lines).strip()
    cleaned = _ANSWER_LABEL_PATTERN.sub("", cleaned, count=1)
    cleaned = strip_citations(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

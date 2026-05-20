"""Post-generation checks for answer accuracy."""

from __future__ import annotations

import re

from src.retrieval.catalog import TranscriptCatalog
from src.retrieval.query_router import QueryIntent

_MEETING_COUNT_RE = re.compile(
    r"\b(\d+)\s+meetings?\b",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def validate_answer(
    answer_text: str,
    catalog: TranscriptCatalog,
    intent: QueryIntent | None,
    source_transcript_ids: set[str],
) -> list[str]:
    """Return user-visible warnings when the answer may be inaccurate."""
    warnings: list[str] = []
    expected = catalog.count()

    if expected == 0:
        return warnings

    count_match = _MEETING_COUNT_RE.search(answer_text or "")
    if count_match and intent in (
        QueryIntent.INVENTORY,
        QueryIntent.PER_MEETING_SUMMARY,
        None,
    ):
        claimed = int(count_match.group(1))
        if claimed != expected:
            warnings.append(
                f"Answer claims {claimed} meetings but the index contains "
                f"{expected}. Prefer the INDEXED MEETINGS count."
            )

    catalog_dates = set(catalog.unique_dates())
    mentioned_dates = set(_ISO_DATE_RE.findall(answer_text or ""))
    unknown_dates = mentioned_dates - catalog_dates
    if unknown_dates:
        warnings.append(
            f"Answer mentions date(s) not in the index: {', '.join(sorted(unknown_dates))}."
        )

    if intent == QueryIntent.PER_MEETING_SUMMARY and source_transcript_ids:
        missing = expected - len(source_transcript_ids)
        if missing > 0:
            warnings.append(
                f"Retrieved excerpts cover {len(source_transcript_ids)} of "
                f"{expected} indexed meetings; some meetings may be under-represented."
            )

    return warnings

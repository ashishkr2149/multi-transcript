"""Heuristic query intent routing for retrieval strategy selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.config import (
    INVENTORY_TOP_K,
    MAX_CHUNKS_PER_TRANSCRIPT,
    PER_MEETING_CHUNKS,
    TOP_K_RESULTS,
)
from src.generation.prompts import wants_speaker_attribution
from src.retrieval.catalog import TranscriptCatalog
from src.utils.helpers import parse_meeting_date

CatalogMode = Literal["full", "summary", "none"]


class QueryIntent(str, Enum):
    INVENTORY = "inventory"
    PER_MEETING_SUMMARY = "per_meeting_summary"
    CROSS_MEETING_SYNTHESIS = "cross_meeting_synthesis"
    SPECIFIC_FACT = "specific_fact"
    ATTRIBUTION = "attribution"


_INVENTORY_RE = re.compile(
    r"\b(how many|number of|count of|list all|list the|which meetings|what meetings|"
    r"what dates|which dates|all meetings|all transcripts|indexed meetings|"
    r"meetings (do we|are|were)|transcripts (do we|are|were))\b.*\b(meeting|transcript|date)s?\b"
    r"|"
    r"\b(meeting|transcript)s?\b.*\b(how many|list all|what dates|which dates)\b",
    re.IGNORECASE,
)

_PER_MEETING_RE = re.compile(
    r"\b(each meeting|every meeting|per meeting|separately|one (summary|section) per|"
    r"summary of each|summarize each|summarise each|individual meeting)\b",
    re.IGNORECASE,
)

_CROSS_MEETING_RE = re.compile(
    r"\b(across (all )?meetings|all meetings|cross[- ]?meeting|compare|combined|"
    r"overall|between meetings|multiple meetings)\b",
    re.IGNORECASE,
)


@dataclass
class QueryPlan:
    intent: QueryIntent
    top_k: int
    max_chunks_per_transcript: int | None
    include_catalog: bool
    catalog_mode: CatalogMode
    filter_meeting_date: str | None
    per_meeting_chunks: int | None = None
    use_per_transcript_retrieval: bool = False


class QueryRouter:
    @staticmethod
    def plan(question: str, catalog: TranscriptCatalog | None = None) -> QueryPlan:
        q = (question or "").strip()
        catalog = catalog or TranscriptCatalog()
        meeting_count = catalog.count()

        if wants_speaker_attribution(q):
            return QueryPlan(
                intent=QueryIntent.ATTRIBUTION,
                top_k=TOP_K_RESULTS,
                max_chunks_per_transcript=MAX_CHUNKS_PER_TRANSCRIPT,
                include_catalog=True,
                catalog_mode="summary",
                filter_meeting_date=_extract_date_filter(q),
            )

        if _INVENTORY_RE.search(q) and not _PER_MEETING_RE.search(q):
            return QueryPlan(
                intent=QueryIntent.INVENTORY,
                top_k=INVENTORY_TOP_K,
                max_chunks_per_transcript=None,
                include_catalog=True,
                catalog_mode="full",
                filter_meeting_date=None,
            )

        if _PER_MEETING_RE.search(q) or (
            _INVENTORY_RE.search(q) and re.search(r"\bsummar", q, re.IGNORECASE)
        ):
            return QueryPlan(
                intent=QueryIntent.PER_MEETING_SUMMARY,
                top_k=TOP_K_RESULTS,
                max_chunks_per_transcript=None,
                include_catalog=True,
                catalog_mode="full",
                filter_meeting_date=_extract_date_filter(q),
                per_meeting_chunks=PER_MEETING_CHUNKS,
                use_per_transcript_retrieval=meeting_count > 0,
            )

        if _CROSS_MEETING_RE.search(q):
            return QueryPlan(
                intent=QueryIntent.CROSS_MEETING_SYNTHESIS,
                top_k=TOP_K_RESULTS,
                max_chunks_per_transcript=MAX_CHUNKS_PER_TRANSCRIPT,
                include_catalog=True,
                catalog_mode="summary",
                filter_meeting_date=_extract_date_filter(q),
            )

        date_filter = _extract_date_filter(q)
        return QueryPlan(
            intent=QueryIntent.SPECIFIC_FACT,
            top_k=TOP_K_RESULTS,
            max_chunks_per_transcript=MAX_CHUNKS_PER_TRANSCRIPT,
            include_catalog=True,
            catalog_mode="summary",
            filter_meeting_date=date_filter,
        )


def _extract_date_filter(question: str) -> str | None:
    """Try to pull an ISO date from the question for Chroma metadata filter."""
    return parse_meeting_date(question)

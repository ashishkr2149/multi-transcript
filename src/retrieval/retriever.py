"""Multi-transcript retriever.

Performs cross-transcript semantic search and groups the resulting chunks by
their source meeting so they can be presented coherently to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import (
    PER_MEETING_CHUNKS,
    RETRIEVAL_OVERSAMPLE,
    TOP_K_RESULTS,
)
from src.retrieval.catalog import TranscriptCatalog
from src.retrieval.query_router import QueryIntent, QueryPlan
from src.retrieval.vector_store import TranscriptVectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    transcript_id: str
    meeting_title: str
    meeting_date: str
    time_range: str
    speakers_in_chunk: list[str]
    primary_speaker: str
    text: str
    similarity: float | None
    has_action_item: bool
    has_screen_share: bool


def _to_retrieved(raw: dict[str, Any]) -> RetrievedChunk:
    meta = raw.get("metadata") or {}
    speakers = [s.strip() for s in (meta.get("speakers_in_chunk") or "").split(",") if s.strip()]
    return RetrievedChunk(
        chunk_id=raw["chunk_id"],
        transcript_id=meta.get("transcript_id", ""),
        meeting_title=meta.get("meeting_title", "") or "",
        meeting_date=meta.get("meeting_date", "") or "",
        time_range=meta.get("time_range", "") or "",
        speakers_in_chunk=speakers,
        primary_speaker=meta.get("primary_speaker", "") or "",
        text=raw["text"],
        similarity=raw.get("similarity"),
        has_action_item=bool(meta.get("has_action_item", False)),
        has_screen_share=bool(meta.get("has_screen_share", False)),
    )


class Retriever:
    def __init__(self, store: TranscriptVectorStore | None = None) -> None:
        self._store = store or TranscriptVectorStore()

    @property
    def store(self) -> TranscriptVectorStore:
        return self._store

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        raw_results = self._store.query(question=question, top_k=top_k, where=where)
        return [_to_retrieved(r) for r in raw_results]

    def retrieve_diversified(
        self,
        question: str,
        plan: QueryPlan,
    ) -> list[RetrievedChunk]:
        """Semantic search with per-transcript caps for broader coverage."""
        if plan.intent == QueryIntent.INVENTORY or plan.top_k <= 0:
            return []

        where: dict[str, Any] | None = None
        if plan.filter_meeting_date:
            where = {"meeting_date": plan.filter_meeting_date}

        oversample = max(plan.top_k * 2, RETRIEVAL_OVERSAMPLE)
        candidates = self.retrieve(question=question, top_k=oversample, where=where)

        if not plan.max_chunks_per_transcript:
            return candidates[: plan.top_k]

        per_tid: dict[str, int] = {}
        selected: list[RetrievedChunk] = []
        for chunk in candidates:
            tid = chunk.transcript_id
            if not tid:
                continue
            count = per_tid.get(tid, 0)
            if count >= plan.max_chunks_per_transcript:
                continue
            selected.append(chunk)
            per_tid[tid] = count + 1
            if len(selected) >= plan.top_k:
                break

        if plan.intent == QueryIntent.PER_MEETING_SUMMARY:
            selected.sort(key=lambda c: c.meeting_date or c.transcript_id)
        return selected

    def retrieve_per_transcript(
        self,
        question: str,
        catalog: TranscriptCatalog,
        chunks_per_meeting: int = PER_MEETING_CHUNKS,
        filter_meeting_date: str | None = None,
    ) -> list[RetrievedChunk]:
        """Fetch top chunks from every indexed transcript (guaranteed coverage)."""
        all_chunks: list[RetrievedChunk] = []
        for rec in catalog.list_all():
            if filter_meeting_date and rec.meeting_date != filter_meeting_date:
                continue
            where = {"transcript_id": rec.transcript_id}
            hits = self.retrieve(
                question=question,
                top_k=chunks_per_meeting,
                where=where,
            )
            all_chunks.extend(hits)
        all_chunks.sort(key=lambda c: (c.meeting_date or "", c.transcript_id))
        return all_chunks

    def retrieve_for_plan(
        self,
        question: str,
        plan: QueryPlan,
        catalog: TranscriptCatalog | None = None,
    ) -> list[RetrievedChunk]:
        """Dispatch retrieval based on query plan."""
        catalog = catalog or TranscriptCatalog(store=self._store)

        if plan.intent == QueryIntent.INVENTORY:
            return []

        if plan.use_per_transcript_retrieval:
            return self.retrieve_per_transcript(
                question=question,
                catalog=catalog,
                chunks_per_meeting=plan.per_meeting_chunks or PER_MEETING_CHUNKS,
                filter_meeting_date=plan.filter_meeting_date,
            )

        return self.retrieve_diversified(question=question, plan=plan)

    @staticmethod
    def group_by_transcript(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
        """Group retrieved chunks by transcript_id preserving rank order."""
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.transcript_id, []).append(chunk)
        return grouped

    @staticmethod
    def format_context(
        chunks: list[RetrievedChunk],
        catalog_block: str | None = None,
    ) -> str:
        """Build context for the LLM with catalog + grouped excerpts."""
        parts: list[str] = []

        if catalog_block:
            parts.append(catalog_block.strip())
            parts.append("")

        if not chunks:
            if catalog_block:
                parts.append(
                    "MEETING EXCERPTS (evidence for what was said)\n"
                    "---------------------------------------------------\n"
                    "(No additional excerpts retrieved — use INDEXED MEETINGS for counts and dates.)"
                )
                return "\n".join(parts).strip()
            return "(No relevant excerpts retrieved.)"

        parts.append("MEETING EXCERPTS (evidence — use for what was said)")
        parts.append("---------------------------------------------------")

        grouped = Retriever.group_by_transcript(chunks)

        def _sort_key(item: tuple[str, list[RetrievedChunk]]) -> str:
            return item[1][0].meeting_date or item[0]

        excerpt_parts: list[str] = []
        for tid, group in sorted(grouped.items(), key=_sort_key):
            head = group[0]
            title = head.meeting_title or tid
            date_suffix = f" ({head.meeting_date})" if head.meeting_date else ""
            excerpt_parts.append(
                f"=== Meeting: {title}{date_suffix} | transcript_id: {tid} ==="
            )
            for idx, chunk in enumerate(group, start=1):
                excerpt_parts.append(
                    f"[Excerpt {idx} | chunk_id: {chunk.chunk_id} | {chunk.time_range} | "
                    f"speakers: {', '.join(chunk.speakers_in_chunk)}]"
                )
                excerpt_parts.append(chunk.text)
                excerpt_parts.append("")

        parts.append("\n".join(excerpt_parts).strip())
        return "\n".join(parts).strip()

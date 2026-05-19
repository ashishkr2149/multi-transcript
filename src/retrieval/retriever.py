"""Multi-transcript retriever.

Performs cross-transcript semantic search and groups the resulting chunks by
their source meeting so they can be presented coherently to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import TOP_K_RESULTS
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

    @staticmethod
    def group_by_transcript(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
        """Group retrieved chunks by transcript_id preserving rank order."""
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.transcript_id, []).append(chunk)
        return grouped

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Build a single context string passed to the LLM.

        Chunks are grouped by meeting so the LLM can clearly see which facts
        come from which transcript. Speakers and timestamps are kept inline.
        """
        if not chunks:
            return "(No relevant excerpts retrieved.)"
        grouped = Retriever.group_by_transcript(chunks)

        def _sort_key(item: tuple[str, list[RetrievedChunk]]) -> str:
            return item[1][0].meeting_date or item[0]

        parts: list[str] = []
        for tid, group in sorted(grouped.items(), key=_sort_key):
            head = group[0]
            header = (
                f"=== Meeting: {head.meeting_title or tid}"
                f"{' (' + head.meeting_date + ')' if head.meeting_date else ''} ==="
            )
            parts.append(header)
            for chunk in group:
                parts.append(
                    f"[Excerpt - {chunk.time_range} | speakers: "
                    f"{', '.join(chunk.speakers_in_chunk)}]"
                )
                parts.append(chunk.text)
                parts.append("")
        return "\n".join(parts).strip()

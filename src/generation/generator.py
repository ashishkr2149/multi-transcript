"""Wraps the retrieve -> prompt -> generate flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY, assert_api_key
from src.generation.prompts import build_messages
from src.generation.validator import validate_answer
from src.retrieval.catalog import TranscriptCatalog
from src.retrieval.query_router import QueryIntent, QueryPlan, QueryRouter
from src.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class AnswerSource:
    chunk_id: str
    transcript_id: str
    meeting_title: str
    meeting_date: str
    time_range: str
    speakers: list[str]
    primary_speaker: str
    similarity: float | None
    text: str


@dataclass
class Answer:
    text: str
    mode: str
    sources: list[AnswerSource]
    used_chunk_ids: list[str]
    warnings: list[str] = field(default_factory=list)
    intent: str = ""
    catalog_count: int = 0


def _to_source(chunk: RetrievedChunk) -> AnswerSource:
    return AnswerSource(
        chunk_id=chunk.chunk_id,
        transcript_id=chunk.transcript_id,
        meeting_title=chunk.meeting_title,
        meeting_date=chunk.meeting_date,
        time_range=chunk.time_range,
        speakers=chunk.speakers_in_chunk,
        primary_speaker=chunk.primary_speaker,
        similarity=chunk.similarity,
        text=chunk.text,
    )


class AnswerGenerator:
    """Glue between the retriever and the OpenAI chat completion API."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        catalog: TranscriptCatalog | None = None,
    ) -> None:
        assert_api_key()
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._retriever = retriever or Retriever()
        self._catalog = catalog or TranscriptCatalog(store=self._retriever.store)

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def catalog(self) -> TranscriptCatalog:
        return self._catalog

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_query: str | None = None,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        force_attribution: bool | None = None,
        temperature: float = 0.2,
    ) -> Answer:
        """Generate an answer with catalog-grounded retrieval."""
        self._catalog.refresh()
        plan = QueryRouter.plan(question, self._catalog)

        if top_k is not None:
            plan.top_k = top_k

        if force_attribution:
            plan.intent = QueryIntent.ATTRIBUTION

        if where and plan.filter_meeting_date is None:
            plan.filter_meeting_date = where.get("meeting_date")

        search_text = retrieval_query or question
        retrieved = self._retriever.retrieve_for_plan(
            question=search_text,
            plan=plan,
            catalog=self._catalog,
        )

        catalog_block = None
        if plan.include_catalog:
            catalog_block = self._catalog.format_for_prompt(plan.catalog_mode)

        context = Retriever.format_context(retrieved, catalog_block=catalog_block)
        history_text = self._format_history(chat_history or [])

        inventory_preface = None
        if plan.intent in (QueryIntent.INVENTORY, QueryIntent.PER_MEETING_SUMMARY):
            inventory_preface = self._catalog.inventory_preface()

        messages, mode = build_messages(
            question=question,
            context=context,
            chat_history_text=history_text,
            force_attribution=force_attribution,
            intent=plan.intent.value,
            inventory_preface=inventory_preface,
        )

        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer_text = (response.choices[0].message.content or "").strip()

        source_ids = {c.transcript_id for c in retrieved if c.transcript_id}
        warnings = validate_answer(
            answer_text=answer_text,
            catalog=self._catalog,
            intent=plan.intent,
            source_transcript_ids=source_ids,
        )

        sources = [_to_source(c) for c in retrieved]
        return Answer(
            text=answer_text,
            mode=mode,
            sources=sources,
            used_chunk_ids=[c.chunk_id for c in retrieved],
            warnings=warnings,
            intent=plan.intent.value,
            catalog_count=self._catalog.count(),
        )

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role", "user").lower()
            label = "User" if role == "user" else "Assistant"
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{label}: {content}")
        return "\n".join(lines)

"""Wraps the retrieve -> prompt -> generate flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY, TOP_K_RESULTS, assert_api_key
from src.generation.prompts import build_messages
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

    def __init__(self, retriever: Retriever | None = None) -> None:
        assert_api_key()
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._retriever = retriever or Retriever()

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = TOP_K_RESULTS,
        where: dict[str, Any] | None = None,
        force_attribution: bool | None = None,
        temperature: float = 0.2,
    ) -> Answer:
        retrieved = self._retriever.retrieve(question=question, top_k=top_k, where=where)
        context = Retriever.format_context(retrieved)
        history_text = self._format_history(chat_history or [])

        messages, mode = build_messages(
            question=question,
            context=context,
            chat_history_text=history_text,
            force_attribution=force_attribution,
        )

        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer_text = (response.choices[0].message.content or "").strip()

        sources = [_to_source(c) for c in retrieved]
        return Answer(
            text=answer_text,
            mode=mode,
            sources=sources,
            used_chunk_ids=[c.chunk_id for c in retrieved],
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

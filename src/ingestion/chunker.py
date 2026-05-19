"""Smart, speaker-aware chunker.

Groups consecutive utterances into chunks bounded by a token budget. Each
chunk preserves speaker attribution inline in the text and keeps rich
metadata for downstream filtering and citation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import tiktoken

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.parser import ParsedTranscript, Utterance
from src.utils.helpers import unique_preserve_order

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of OpenAI tokens for the given text."""
    if not text:
        return 0
    return len(_ENCODING.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    transcript_id: str
    meeting_title: str | None
    meeting_date: str | None
    time_range: str
    speakers_in_chunk: list[str]
    primary_speaker: str
    has_action_item: bool
    has_screen_share: bool
    text: str
    text_for_embedding: str
    token_count: int

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict suitable for ChromaDB (scalar values only)."""
        return {
            "transcript_id": self.transcript_id,
            "meeting_title": self.meeting_title or "",
            "meeting_date": self.meeting_date or "",
            "time_range": self.time_range,
            "speakers_in_chunk": ", ".join(self.speakers_in_chunk),
            "primary_speaker": self.primary_speaker,
            "has_action_item": self.has_action_item,
            "has_screen_share": self.has_screen_share,
            "text": self.text,
            "token_count": self.token_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_utterance_for_text(u: Utterance) -> str:
    return f"[{u.speaker} @ {u.timestamp}]: {u.text}"


def _build_chunk(
    transcript: ParsedTranscript,
    utterances: list[Utterance],
    chunk_index: int,
) -> Chunk:
    speakers = [u.speaker for u in utterances]
    speakers_unique = unique_preserve_order(speakers)
    primary_speaker = Counter(speakers).most_common(1)[0][0] if speakers else "Unknown"

    chunk_text = "\n".join(_format_utterance_for_text(u) for u in utterances)
    embed_header_parts: list[str] = []
    if transcript.meeting_title:
        embed_header_parts.append(f"Meeting: {transcript.meeting_title}")
    if transcript.meeting_date:
        embed_header_parts.append(f"Date: {transcript.meeting_date}")
    embed_header_parts.append(f"Speakers: {', '.join(speakers_unique)}")
    embed_header = ". ".join(embed_header_parts) + "."
    text_for_embedding = embed_header + "\n" + chunk_text

    time_range = f"{utterances[0].timestamp} - {utterances[-1].timestamp}"

    return Chunk(
        chunk_id=f"{transcript.transcript_id}_chunk_{chunk_index:04d}",
        transcript_id=transcript.transcript_id,
        meeting_title=transcript.meeting_title,
        meeting_date=transcript.meeting_date,
        time_range=time_range,
        speakers_in_chunk=speakers_unique,
        primary_speaker=primary_speaker,
        has_action_item=any(u.has_action_item for u in utterances),
        has_screen_share=any(u.has_screen_share for u in utterances),
        text=chunk_text,
        text_for_embedding=text_for_embedding,
        token_count=count_tokens(chunk_text),
    )


def chunk_transcript(
    transcript: ParsedTranscript,
    max_tokens: int = CHUNK_SIZE,
    overlap_utterances: int = 1,
) -> list[Chunk]:
    """Chunk a parsed transcript into token-bounded, speaker-preserving chunks.

    The strategy:
      * Walk utterances in order.
      * Accumulate them into the current chunk until adding the next utterance
        would exceed ``max_tokens`` (computed on the to-be-embedded text).
      * Start the next chunk by carrying over ``overlap_utterances`` from the
        tail of the previous chunk so context bridges the boundary.
      * Single utterances larger than ``max_tokens`` are kept whole (we never
        split mid-utterance to preserve speaker attribution and meaning).
    """
    if not transcript.utterances:
        return []

    chunks: list[Chunk] = []
    current: list[Utterance] = []
    current_tokens = 0
    chunk_index = 0

    for utterance in transcript.utterances:
        utt_text = _format_utterance_for_text(utterance)
        utt_tokens = count_tokens(utt_text)

        if current and current_tokens + utt_tokens > max_tokens:
            chunks.append(_build_chunk(transcript, current, chunk_index))
            chunk_index += 1
            if overlap_utterances > 0:
                tail = current[-overlap_utterances:]
                current = list(tail)
                current_tokens = sum(count_tokens(_format_utterance_for_text(u)) for u in current)
            else:
                current = []
                current_tokens = 0

        current.append(utterance)
        current_tokens += utt_tokens

    if current:
        chunks.append(_build_chunk(transcript, current, chunk_index))

    return chunks


__all__ = ["Chunk", "chunk_transcript", "count_tokens", "CHUNK_OVERLAP"]

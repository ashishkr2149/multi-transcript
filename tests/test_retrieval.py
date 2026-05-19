"""Smoke tests for chunking and prompt-mode selection.

These tests don't hit the network. The vector store and OpenAI calls are
exercised manually via ``scripts/ingest_transcripts.py``.
"""

from __future__ import annotations

from src.generation.prompts import wants_speaker_attribution
from src.ingestion.chunker import chunk_transcript, count_tokens
from src.ingestion.parser import parse_transcript_text

SAMPLE = """Sample Meeting - May 04
VIEW RECORDING - 5 mins:

---

0:01 - Alice
  We need to talk about the ETL pipeline.

0:02 - Bob
  Sure. What about it?

0:03 - Alice
  The transformation rules are unclear.

0:04 - Bob
  I will document them this week.
"""


def test_chunker_produces_chunks():
    parsed = parse_transcript_text(SAMPLE, transcript_id="t1", source_file="t1.txt")
    chunks = chunk_transcript(parsed, max_tokens=200)
    assert chunks, "expected at least one chunk"
    first = chunks[0]
    assert first.transcript_id == "t1"
    assert first.primary_speaker in {"Alice", "Bob"}
    assert "[Alice @ 0:01]" in first.text
    assert first.token_count > 0


def test_attribution_detection():
    assert wants_speaker_attribution("Who explained the ETL process?")
    assert wants_speaker_attribution("Which speaker suggested the new flow?")
    assert wants_speaker_attribution("Who is responsible for monitoring?")
    assert not wants_speaker_attribution("What is the ETL process?")
    assert not wants_speaker_attribution("Summarise the discussion about caching")


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


if __name__ == "__main__":
    test_chunker_produces_chunks()
    test_attribution_detection()
    test_count_tokens()
    print("retrieval tests passed")

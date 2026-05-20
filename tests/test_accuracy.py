"""Accuracy framework tests (no network)."""

from __future__ import annotations

from src.ingestion.parser import parse_transcript_text
from src.ingestion.service import metadata_quality_warnings
from src.retrieval.catalog import TranscriptCatalog, title_looks_like_date_only
from src.retrieval.query_router import QueryIntent, QueryRouter
from src.retrieval.retriever import Retriever

GEMINI_SPLIT_LINE_RAW = """Feb 10, 2026
DAILY: PLG, Website, Content, Academy, etc - Transcript
00:00:00

Jonathan Assayag: Hey folks,
Nazima Piracha: Say hi John.
00:01:46
Nazima Piracha: Three consultants still engaged.
Transcription ended after 01:20:05
"""


def test_title_looks_like_date_only():
    assert title_looks_like_date_only("Feb 10, 2026", "2026-02-10")
    assert not title_looks_like_date_only("DAILY: PLG - Transcript", "2026-02-10")


def test_router_inventory_intent():
    plan = QueryRouter.plan("How many meetings were held?")
    assert plan.intent == QueryIntent.INVENTORY
    assert plan.top_k == 0


def test_router_per_meeting_summary_intent():
    plan = QueryRouter.plan(
        "How many meetings were held and give summary of each meeting separately"
    )
    assert plan.intent == QueryIntent.PER_MEETING_SUMMARY
    assert plan.use_per_transcript_retrieval is True


def test_router_cross_meeting_intent():
    plan = QueryRouter.plan("What was discussed about ETL across all meetings?")
    assert plan.intent == QueryIntent.CROSS_MEETING_SYNTHESIS


def test_parse_gemini_title_not_date():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t", "t.txt")
    assert parsed.meeting_title != "Feb 10, 2026"
    assert "PLG" in (parsed.meeting_title or "")
    assert parsed.meeting_date == "2026-02-10"
    assert "DAILY" not in parsed.participants


def test_metadata_quality_warnings_no_false_positive_on_good_parse():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t", "t.txt")
    warns = metadata_quality_warnings(parsed, GEMINI_SPLIT_LINE_RAW)
    assert not any("title looks like a date" in w for w in warns)


def test_format_context_includes_catalog():
    from src.retrieval.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            transcript_id="t1",
            meeting_title="Meet A",
            meeting_date="2026-02-09",
            time_range="0:00 - 1:00",
            speakers_in_chunk=["Alice"],
            primary_speaker="Alice",
            text="Hello",
            similarity=0.9,
            has_action_item=False,
            has_screen_share=False,
        )
    ]
    catalog_block = "INDEXED MEETINGS\n1. Meet A (2026-02-09) [transcript_id: t1]"
    ctx = Retriever.format_context(chunks, catalog_block=catalog_block)
    assert "INDEXED MEETINGS" in ctx
    assert "transcript_id: t1" in ctx
    assert "MEETING EXCERPTS" in ctx


def test_validator_meeting_count_mismatch():
    from src.generation.validator import validate_answer

    class FakeCatalog:
        def count(self):
            return 5

        def unique_dates(self):
            return ["2026-02-09", "2026-02-10"]

    warnings = validate_answer(
        "A total of 6 meetings were held.",
        FakeCatalog(),
        QueryIntent.INVENTORY,
        set(),
    )
    assert any("claims 6" in w for w in warnings)


if __name__ == "__main__":
    test_title_looks_like_date_only()
    test_router_inventory_intent()
    test_router_per_meeting_summary_intent()
    test_router_cross_meeting_intent()
    test_parse_gemini_title_not_date()
    test_metadata_quality_warnings_no_false_positive_on_good_parse()
    test_format_context_includes_catalog()
    test_validator_meeting_count_mismatch()
    print("accuracy tests passed")

"""Lightweight checks for the transcript parser."""

from __future__ import annotations

from src.ingestion.parser import parse_transcript_text

SAMPLE = """Impromptu Zoom Meeting - May 04
VIEW RECORDING - 24 mins (No highlights):

---

0:09 - Andrew Gaikwad (kainyx.com)
  Can you hear me okay?

0:10 - Learner Park Media
  Yes, I can hear you.
  And one more line.

0:12 - Andrew Gaikwad (kainyx.com)
  Awesome.
  ACTION ITEM: Send Andrew the recording link - WATCH: https://example.com/123
"""


def test_parses_header_and_duration():
    parsed = parse_transcript_text(SAMPLE, transcript_id="t1", source_file="t1.txt")
    assert parsed.meeting_title == "Impromptu Zoom Meeting - May 04"
    assert parsed.duration_minutes == 24
    assert parsed.meeting_date == "2026-05-04" or parsed.meeting_date.endswith("-05-04")


def test_parses_utterances_and_participants():
    parsed = parse_transcript_text(SAMPLE, transcript_id="t1", source_file="t1.txt")
    assert len(parsed.utterances) == 3
    assert parsed.utterances[0].speaker == "Andrew Gaikwad"
    assert parsed.utterances[0].speaker_org == "kainyx.com"
    assert "more line" in parsed.utterances[1].text
    assert parsed.participants == ["Andrew Gaikwad", "Learner Park Media"]


def test_extracts_action_items():
    parsed = parse_transcript_text(SAMPLE, transcript_id="t1", source_file="t1.txt")
    assert len(parsed.action_items) == 1
    assert parsed.action_items[0]["speaker"] == "Andrew Gaikwad"
    assert "recording link" in parsed.action_items[0]["description"]


if __name__ == "__main__":
    test_parses_header_and_duration()
    test_parses_utterances_and_participants()
    test_extracts_action_items()
    print("parser tests passed")

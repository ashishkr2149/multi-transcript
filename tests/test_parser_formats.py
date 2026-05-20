"""Tests for multi-format transcript detection and parsing."""

from __future__ import annotations

from src.ingestion.formats import TranscriptFormat, detect_format
from src.ingestion.parser import parse_transcript_text

FATHOM_SAMPLE = """Impromptu Zoom Meeting - May 04
VIEW RECORDING - 24 mins (No highlights):

---

0:09 - Andrew Gaikwad (kainyx.com)
  Can you hear me okay?

0:10 - Learner Park Media
  Yes, I can hear you.
"""

GEMINI_SPLIT_LINE_SAMPLE = """📖 Transcript
Feb 9, 2026
DAILY: PLG - Transcript
00:00:00
Nazima Piracha: Okay. Should I start with the update?
Jonathan Assayag: Yeah, sounds good.
00:01:46
Nazima Piracha: Three consultants still engaged.
Transcription ended after 01:20:05
"""

# Real-world pattern from pasted Google Docs (no emoji header, split timestamps)
GEMINI_SPLIT_LINE_RAW = """Feb 10, 2026
DAILY: PLG, Website, Content, Academy, etc - Transcript
00:00:00

Jonathan Assayag: Hey folks,
Nazima Piracha: Say hi John.
00:01:46
Nazima Piracha: Three consultants still engaged.
Transcription ended after 01:20:05
"""

GEMINI_SAMPLE = """DAILY: PLG, Website, Content, Academy, etc - Transcript
Feb 9, 2026

00:00:00 Nazima Piracha: Okay. Should I start with the update?
00:00:53 Jonathan Assayag: Yeah, sounds good.
00:01:46 Nazima Piracha: So we have three consultants still in the engagement.
Transcription ended after 01:20:05
"""

ZOOM_VTT_SAMPLE = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
Jonathan Assayag: Hello everyone.

2
00:00:05.000 --> 00:00:08.000
Nazima Piracha: Thanks for joining.
"""

OTTER_SAMPLE = """Team Standup - Feb 9

Jonathan Assayag  0:00
Hello everyone, let's begin.

Nazima Piracha  0:15
Sounds good to me.
"""

PLAIN_SAMPLE = """Project Kickoff

Alice: Welcome everyone to the kickoff.
Bob: Thanks Alice, excited to be here.
"""

PLAIN_NO_TITLE = """Alice: Welcome everyone.
Bob: Thanks Alice.
"""


def test_detect_fathom():
    assert detect_format(FATHOM_SAMPLE) == TranscriptFormat.FATHOM


def test_detect_gemini():
    assert detect_format(GEMINI_SAMPLE) == TranscriptFormat.GEMINI


def test_detect_gemini_split_line_raw():
    assert detect_format(GEMINI_SPLIT_LINE_RAW) == TranscriptFormat.GEMINI


def test_detect_zoom_vtt():
    assert detect_format(ZOOM_VTT_SAMPLE) == TranscriptFormat.ZOOM_VTT


def test_detect_otter():
    assert detect_format(OTTER_SAMPLE) == TranscriptFormat.OTTER


def test_detect_plain_dialogue():
    assert detect_format(PLAIN_SAMPLE) == TranscriptFormat.PLAIN_DIALOGUE


def test_parse_fathom_regression():
    parsed = parse_transcript_text(FATHOM_SAMPLE, "t1", "t1.txt")
    assert parsed.detected_format == "fathom"
    assert len(parsed.utterances) == 2
    assert parsed.utterances[0].speaker == "Andrew Gaikwad"


def test_parse_gemini_split_line_format():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_SAMPLE, "t2b", "t2b.txt")
    assert parsed.detected_format == "gemini"
    assert len(parsed.utterances) >= 3
    assert parsed.utterances[0].speaker == "Nazima Piracha"
    assert parsed.normalization_applied is True


def test_parse_gemini_split_line_raw_without_manual_normalize():
    """Pasted txt should parse correctly without manual normalization."""
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_raw", "t_raw.txt")
    assert parsed.detected_format == "gemini"
    assert parsed.meeting_title == "DAILY: PLG, Website, Content, Academy, etc - Transcript"
    assert parsed.meeting_date == "2026-02-10"
    assert parsed.duration_minutes is not None
    assert parsed.duration_minutes >= 60
    assert "DAILY" not in parsed.participants
    assert parsed.utterances[0].speaker == "Jonathan Assayag"
    assert parsed.utterances[0].timestamp == "00:00:00"
    assert parsed.utterances[0].timestamp_seconds == 0


def test_parse_gemini_correct_title_not_date():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_title", "t_title.txt")
    assert parsed.meeting_title != "Feb 10, 2026"
    assert "PLG" in (parsed.meeting_title or "")


def test_parse_gemini_correct_date():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_date", "t_date.txt")
    assert parsed.meeting_date == "2026-02-10"


def test_parse_gemini_duration_from_footer():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_dur", "t_dur.txt")
    assert parsed.duration_minutes == 80


def test_title_not_speaker():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_spk", "t_spk.txt")
    speakers = {u.speaker for u in parsed.utterances}
    assert "DAILY" not in speakers


def test_parse_gemini_utterances():
    parsed = parse_transcript_text(GEMINI_SAMPLE, "t2", "t2.txt")
    assert parsed.detected_format == "gemini"
    assert len(parsed.utterances) >= 3
    assert parsed.utterances[0].speaker == "Nazima Piracha"
    assert "update" in parsed.utterances[0].text.lower()
    assert parsed.utterances[0].timestamp_seconds == 0
    assert parsed.utterances[1].timestamp_seconds == 53


def test_parse_gemini_duration():
    parsed = parse_transcript_text(GEMINI_SAMPLE, "t2", "t2.txt")
    assert parsed.duration_minutes is not None
    assert parsed.duration_minutes >= 60


def test_parse_zoom_vtt():
    parsed = parse_transcript_text(ZOOM_VTT_SAMPLE, "t3", "t3.txt")
    assert parsed.detected_format == "zoom_vtt"
    assert len(parsed.utterances) == 2
    assert "Hello" in parsed.utterances[0].text


def test_parse_otter():
    parsed = parse_transcript_text(OTTER_SAMPLE, "t4", "t4.txt")
    assert parsed.detected_format == "otter"
    assert len(parsed.utterances) == 2
    assert parsed.utterances[0].speaker == "Jonathan Assayag"


def test_parse_plain_dialogue():
    parsed = parse_transcript_text(PLAIN_SAMPLE, "t5", "t5.txt")
    assert parsed.detected_format == "plain_dialogue"
    assert len(parsed.utterances) == 2
    assert parsed.utterances[0].speaker == "Alice"
    assert parsed.meeting_title == "Project Kickoff"


def test_missing_timestamps_graceful():
    parsed = parse_transcript_text(PLAIN_NO_TITLE, "t6", "t6.txt")
    assert len(parsed.utterances) == 2
    assert parsed.utterances[0].timestamp.startswith("line_")


def test_universal_metadata_extraction():
    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_meta", "t_meta.txt")
    assert parsed.meeting_date == "2026-02-10"
    assert parsed.meeting_title is not None
    assert "metadata" in parsed.metadata_source or "header" in parsed.metadata_source or "parser" in parsed.metadata_source


def test_end_to_end_paste_flow():
    from src.ingestion.chunker import chunk_transcript

    parsed = parse_transcript_text(GEMINI_SPLIT_LINE_RAW, "t_e2e", "t_e2e.txt")
    chunks = chunk_transcript(parsed)
    assert parsed.detected_format == "gemini"
    assert len(chunks) >= 1
    assert chunks[0].meeting_date == "2026-02-10"
    assert "PLG" in (chunks[0].meeting_title or "")


def test_chunking_after_gemini_parse():
    from src.ingestion.chunker import chunk_transcript

    parsed = parse_transcript_text(GEMINI_SAMPLE, "t2", "t2.txt")
    chunks = chunk_transcript(parsed)
    assert len(chunks) >= 1
    assert chunks[0].transcript_id == "t2"


if __name__ == "__main__":
    test_detect_fathom()
    test_detect_gemini()
    test_detect_gemini_split_line_raw()
    test_detect_zoom_vtt()
    test_detect_otter()
    test_detect_plain_dialogue()
    test_parse_fathom_regression()
    test_parse_gemini_split_line_format()
    test_parse_gemini_split_line_raw_without_manual_normalize()
    test_parse_gemini_correct_title_not_date()
    test_parse_gemini_correct_date()
    test_parse_gemini_duration_from_footer()
    test_title_not_speaker()
    test_parse_gemini_utterances()
    test_parse_gemini_duration()
    test_parse_zoom_vtt()
    test_parse_otter()
    test_parse_plain_dialogue()
    test_missing_timestamps_graceful()
    test_universal_metadata_extraction()
    test_end_to_end_paste_flow()
    test_chunking_after_gemini_parse()
    print("all format parser tests passed")

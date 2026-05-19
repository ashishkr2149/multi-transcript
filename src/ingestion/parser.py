"""Transcript parser.

Converts raw transcript text files (Fathom-style export) into a structured
JSON document with speakers, timestamps, action items and screen-share events.

The expected raw format looks like:

    Impromptu Zoom Meeting - May 04
    VIEW RECORDING - 24 mins (No highlights):

    ---

    0:09 - Andrew Gaikwad (kainyx.com)
      Can you hear me okay?

    0:10 - Learner Park Media
      Yes, I can hear you.
      ...

Multi-line utterances are supported (lines indented with whitespace continue
the previous speaker's text). Special markers like ``SCREEN SHARING:`` and
``ACTION ITEM:`` are extracted into their own metadata buckets while still
remaining attached to the surrounding utterance for context.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.helpers import (
    normalize_speaker_name,
    parse_meeting_date,
    timestamp_to_seconds,
    unique_preserve_order,
)

SPEAKER_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<name>[^()]+?)(?:\s*\((?P<org>[^)]+)\))?\s*$"
)

DURATION_RE = re.compile(r"VIEW\s+RECORDING.*?-\s*(\d+)\s*mins", re.IGNORECASE)

ACTION_ITEM_RE = re.compile(r"ACTION ITEM:\s*(.*?)(?=\s*-\s*WATCH:|$)", re.IGNORECASE)
SCREEN_SHARE_RE = re.compile(r"SCREEN SHARING:\s*(.*?)(?=\s*-\s*WATCH:|$)", re.IGNORECASE)
WATCH_LINK_RE = re.compile(r"WATCH:\s*(https?://\S+)")


@dataclass
class Utterance:
    timestamp: str
    timestamp_seconds: int
    speaker: str
    speaker_org: str | None
    text: str
    has_action_item: bool = False
    has_screen_share: bool = False


@dataclass
class ParsedTranscript:
    transcript_id: str
    source_file: str
    meeting_title: str | None
    meeting_date: str | None
    duration_minutes: int | None
    participants: list[str]
    utterances: list[Utterance] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    screen_shares: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def _clean_utterance_text(text: str) -> str:
    """Strip embedded marker links/text from raw utterance content."""
    cleaned = WATCH_LINK_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_transcript_text(raw: str, transcript_id: str, source_file: str) -> ParsedTranscript:
    """Parse a raw transcript string into a :class:`ParsedTranscript`."""
    lines = raw.splitlines()
    title: str | None = None
    duration: int | None = None
    body_start = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if title is None and not stripped.startswith("VIEW RECORDING") and stripped != "---":
            title = stripped
            continue
        dur_match = DURATION_RE.search(stripped)
        if dur_match:
            duration = int(dur_match.group(1))
        if stripped == "---":
            body_start = idx + 1
            break

    utterances: list[Utterance] = []
    action_items: list[dict[str, Any]] = []
    screen_shares: list[dict[str, Any]] = []
    current: Utterance | None = None
    pending_text_parts: list[str] = []

    def _flush() -> None:
        nonlocal current, pending_text_parts
        if current is None:
            return
        full_text = " ".join(pending_text_parts).strip()
        for ai_match in ACTION_ITEM_RE.finditer(full_text):
            action_items.append(
                {
                    "timestamp": current.timestamp,
                    "speaker": current.speaker,
                    "description": ai_match.group(1).strip(" .-"),
                }
            )
            current.has_action_item = True
        for ss_match in SCREEN_SHARE_RE.finditer(full_text):
            screen_shares.append(
                {
                    "timestamp": current.timestamp,
                    "speaker": current.speaker,
                    "description": ss_match.group(1).strip(" .-"),
                }
            )
            current.has_screen_share = True
        current.text = _clean_utterance_text(
            ACTION_ITEM_RE.sub("", SCREEN_SHARE_RE.sub("", full_text))
        )
        if current.text:
            utterances.append(current)
        current = None
        pending_text_parts = []

    for line in lines[body_start:]:
        if not line.strip():
            continue
        speaker_match = SPEAKER_LINE_RE.match(line.strip())
        if speaker_match and not line.startswith(" "):
            _flush()
            timestamp = speaker_match.group("timestamp")
            name = normalize_speaker_name(speaker_match.group("name"))
            org = speaker_match.group("org")
            current = Utterance(
                timestamp=timestamp,
                timestamp_seconds=timestamp_to_seconds(timestamp),
                speaker=name,
                speaker_org=org,
                text="",
            )
        else:
            if current is None:
                continue
            pending_text_parts.append(line.strip())

    _flush()

    participants = unique_preserve_order(u.speaker for u in utterances)
    meeting_date = parse_meeting_date(title or "")

    return ParsedTranscript(
        transcript_id=transcript_id,
        source_file=source_file,
        meeting_title=title,
        meeting_date=meeting_date,
        duration_minutes=duration,
        participants=participants,
        utterances=utterances,
        action_items=action_items,
        screen_shares=screen_shares,
    )


def parse_transcript_file(path: Path, transcript_id: str | None = None) -> ParsedTranscript:
    """Read a transcript text file from disk and return its parsed form."""
    raw = path.read_text(encoding="utf-8")
    return parse_transcript_text(
        raw=raw,
        transcript_id=transcript_id or path.stem,
        source_file=path.name,
    )


def save_parsed_transcript(parsed: ParsedTranscript, out_path: Path) -> None:
    """Persist a :class:`ParsedTranscript` to disk as pretty JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(parsed.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

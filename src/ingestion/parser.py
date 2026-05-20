"""Multi-format transcript parser.

Auto-detects export format (Fathom, Gemini/Google Meet, Zoom VTT, Otter,
plain dialogue) and normalizes all inputs into the same :class:`ParsedTranscript`
structure for chunking and retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ingestion.extractors import prepare_text_for_parsing
from src.ingestion.formats import TranscriptFormat, detect_format
from src.utils.helpers import (
    normalize_speaker_name,
    parse_meeting_date,
    timestamp_to_seconds,
    unique_preserve_order,
)

# --- Fathom ---
SPEAKER_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<name>[^()]+?)(?:\s*\((?P<org>[^)]+)\))?\s*$"
)
DURATION_RE = re.compile(r"VIEW\s+RECORDING.*?-\s*(\d+)\s*mins", re.IGNORECASE)
ACTION_ITEM_RE = re.compile(r"ACTION ITEM:\s*(.*?)(?=\s*-\s*WATCH:|$)", re.IGNORECASE)
SCREEN_SHARE_RE = re.compile(r"SCREEN SHARING:\s*(.*?)(?=\s*-\s*WATCH:|$)", re.IGNORECASE)
WATCH_LINK_RE = re.compile(r"WATCH:\s*(https?://\S+)")

# --- Gemini / Google Meet ---
GEMINI_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{1,2}:\d{2}:\d{2})\s+"
    r"(?P<name>[^:]+?):\s*"
    r"(?P<text>.*)$"
)
GEMINI_TIMESTAMP_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
GEMINI_SPEAKER_LINE_RE = re.compile(r"^(?P<name>[^:]+?):\s*(?P<text>.+)$")
GEMINI_DURATION_RE = re.compile(
    r"Transcription ended after\s+(\d{1,2}):(\d{2}):(\d{2})",
    re.IGNORECASE,
)
GEMINI_DATE_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+)?(\d{1,2}),\s*(\d{4})$"
)
_STANDALONE_DATE_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+)?\d{1,2},?\s*\d{4}$"
)
_TITLE_KEYWORDS_RE = re.compile(
    r"\b(transcript|meeting|call|standup|sync|daily|zoom|teams|kickoff|review)\b",
    re.IGNORECASE,
)

# --- Zoom VTT ---
ZOOM_VTT_TIME_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\s*$"
)

# --- Otter ---
OTTER_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z\s]{1,50}?)\s+"
    r"(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)

# --- Plain dialogue ---
PLAIN_DIALOGUE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z\s]{1,50}?):\s*(?P<text>.+)$"
)

# Skip boilerplate lines common in Gemini exports
_GEMINI_SKIP_RE = re.compile(
    r"^(📝|📖|Notes|Transcript|Meeting records|Attachments|"
    r"Please provide feedback|Get tips|Suggested next steps|"
    r"You should review|This editable transcript|Invited |"
    r"Transcript Summary|Summary$)",
    re.IGNORECASE,
)


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
    detected_format: str = "unknown"
    normalization_applied: bool = False
    metadata_source: str = "parser"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_utterance_text(text: str) -> str:
    cleaned = WATCH_LINK_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_standalone_date_line(line: str) -> bool:
    """True when the line is only a calendar date (not a meeting title)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(GEMINI_DATE_RE.match(stripped)) or bool(_STANDALONE_DATE_RE.match(stripped))


def _is_likely_title_line(line: str) -> bool:
    """True when the line looks like a meeting title, not a speaker utterance."""
    stripped = line.strip()
    if not stripped or _is_standalone_date_line(stripped):
        return False
    if GEMINI_TIMESTAMP_ONLY_RE.match(stripped):
        return False
    if GEMINI_LINE_RE.match(stripped):
        return False
    if OTTER_LINE_RE.match(stripped):
        return False
    if SPEAKER_LINE_RE.match(stripped):
        return False
    dialogue_match = PLAIN_DIALOGUE_RE.match(stripped)
    if dialogue_match:
        name = dialogue_match.group("name").strip()
        text = dialogue_match.group("text").strip()
        # Short speaker names without meeting keywords are utterances, not titles.
        if (
            len(name.split()) <= 4
            and not _TITLE_KEYWORDS_RE.search(name)
            and "transcript" not in text.lower()
            and " - " not in text
        ):
            return False
    lower = stripped.lower()
    if "transcript" in lower and (" - " in stripped or ":" in stripped):
        return True
    if _TITLE_KEYWORDS_RE.search(stripped) and len(stripped) > 12:
        return True
    if " - " in stripped and re.search(r"\d{4}", stripped):
        return True
    return False


def _extract_duration_from_text(text: str) -> int | None:
    for line in text.splitlines():
        dur_match = GEMINI_DURATION_RE.search(line.strip())
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), int(dur_match.group(3))
            total_sec = h * 3600 + m * 60 + s
            return max(1, round(total_sec / 60))
    return None


def _extract_universal_metadata(
    lines: list[str],
    scan_lines: int = 30,
) -> tuple[str | None, str | None, int | None, str]:
    """Extract title, ISO date, and duration from any transcript header/footer.

    Returns ``(title, meeting_date, duration_minutes, metadata_source)``.
    """
    title: str | None = None
    meeting_date: str | None = None
    duration_minutes = _extract_duration_from_text("\n".join(lines))
    sources: list[str] = []

    for stripped in (ln.strip() for ln in lines[:scan_lines] if ln.strip()):
        if _GEMINI_SKIP_RE.match(stripped):
            continue
        if GEMINI_DURATION_RE.search(stripped):
            if duration_minutes is None:
                duration_minutes = _extract_duration_from_text(stripped)
            continue
        if _is_standalone_date_line(stripped) and not meeting_date:
            meeting_date = parse_meeting_date(stripped)
            if meeting_date:
                sources.append("header_date")
            continue
        if _is_likely_title_line(stripped) and not title:
            title = stripped
            sources.append("header_title")

    if not meeting_date and title:
        inferred = parse_meeting_date(title)
        if inferred:
            meeting_date = inferred
            sources.append("title_date")

    if duration_minutes:
        sources.append("footer_duration")

    metadata_source = "+".join(dict.fromkeys(sources)) if sources else "none"
    return title, meeting_date, duration_minutes, metadata_source


def _merge_metadata(
    universal: tuple[str | None, str | None, int | None, str],
    parsed: tuple[str | None, str | None, int | None],
) -> tuple[str | None, str | None, int | None, str]:
    """Prefer format-specific values; fall back to universal extraction."""
    u_title, u_date, u_duration, u_source = universal
    p_title, p_date, p_duration = parsed

    title = p_title or u_title
    meeting_date = p_date or u_date
    duration = p_duration if p_duration is not None else u_duration

    sources: list[str] = []
    if p_title:
        sources.append("parser_title")
    elif u_title:
        sources.append(u_source.split("_")[0] + "_title" if u_source != "none" else "universal_title")

    if p_date:
        sources.append("parser_date")
    elif u_date:
        sources.append("universal_date")

    if p_duration is not None:
        sources.append("parser_duration")
    elif u_duration is not None:
        sources.append("universal_duration")

    metadata_source = "+".join(dict.fromkeys(sources)) if sources else "none"
    return title, meeting_date, duration, metadata_source


def _find_dialogue_body_start(lines: list[str]) -> int:
    """Skip metadata header lines before the first real speaker line."""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_standalone_date_line(stripped) or _is_likely_title_line(stripped):
            continue
        if GEMINI_DURATION_RE.search(stripped) or _GEMINI_SKIP_RE.match(stripped):
            continue
        if GEMINI_TIMESTAMP_ONLY_RE.match(stripped):
            return idx
        if GEMINI_LINE_RE.match(stripped):
            return idx
        if PLAIN_DIALOGUE_RE.match(stripped) and not _is_likely_title_line(stripped):
            return idx
        if OTTER_LINE_RE.match(stripped):
            return idx
        if SPEAKER_LINE_RE.match(stripped):
            return idx
    return 0


def _apply_fathom_markers(
    utterance: Utterance,
    full_text: str,
    action_items: list[dict[str, Any]],
    screen_shares: list[dict[str, Any]],
) -> None:
    for ai_match in ACTION_ITEM_RE.finditer(full_text):
        action_items.append(
            {
                "timestamp": utterance.timestamp,
                "speaker": utterance.speaker,
                "description": ai_match.group(1).strip(" .-"),
            }
        )
        utterance.has_action_item = True
    for ss_match in SCREEN_SHARE_RE.finditer(full_text):
        screen_shares.append(
            {
                "timestamp": utterance.timestamp,
                "speaker": utterance.speaker,
                "description": ss_match.group(1).strip(" .-"),
            }
        )
        utterance.has_screen_share = True
    utterance.text = _clean_utterance_text(
        ACTION_ITEM_RE.sub("", SCREEN_SHARE_RE.sub("", full_text))
    )


def _parse_fathom_body(
    lines: list[str],
    body_start: int,
) -> tuple[list[Utterance], list[dict[str, Any]], list[dict[str, Any]]]:
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
        _apply_fathom_markers(current, full_text, action_items, screen_shares)
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
            current = Utterance(
                timestamp=timestamp,
                timestamp_seconds=timestamp_to_seconds(timestamp),
                speaker=normalize_speaker_name(speaker_match.group("name")),
                speaker_org=speaker_match.group("org"),
                text="",
            )
        elif current is not None:
            pending_text_parts.append(line.strip())

    _flush()
    return utterances, action_items, screen_shares


def _parse_fathom_header(lines: list[str]) -> tuple[str | None, int | None, int]:
    title: str | None = None
    duration: int | None = None
    body_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if title is None and not stripped.startswith("VIEW RECORDING") and stripped != "---":
            if not _GEMINI_SKIP_RE.match(stripped) and not _is_standalone_date_line(stripped):
                title = stripped
            continue
        dur_match = DURATION_RE.search(stripped)
        if dur_match:
            duration = int(dur_match.group(1))
        if stripped == "---":
            body_start = idx + 1
            break
    return title, duration, body_start


def _parse_gemini_metadata(
    lines: list[str],
) -> tuple[str | None, str | None, int | None, int]:
    """Extract title, ISO date, duration minutes, and body start index."""
    title: str | None = None
    meeting_date: str | None = None
    duration_minutes: int | None = None
    body_start = 0
    in_transcript = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if re.search(r"\btranscript\b", stripped, re.IGNORECASE) and not in_transcript:
            in_transcript = True
            if _is_likely_title_line(stripped):
                title = stripped
            body_start = idx + 1
            continue

        dur_match = GEMINI_DURATION_RE.search(stripped)
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), int(dur_match.group(3))
            total_sec = h * 3600 + m * 60 + s
            duration_minutes = max(1, round(total_sec / 60))
            continue

        if _is_standalone_date_line(stripped) and not meeting_date:
            meeting_date = parse_meeting_date(stripped)
            continue

        if GEMINI_LINE_RE.match(stripped):
            if body_start == 0:
                body_start = idx
            continue

        if GEMINI_TIMESTAMP_ONLY_RE.match(stripped):
            if body_start == 0:
                body_start = idx
            continue

        if title is None and _is_likely_title_line(stripped):
            title = stripped

    if body_start == 0:
        body_start = _find_dialogue_body_start(lines)

    return title, meeting_date, duration_minutes, body_start


def _parse_gemini_body(lines: list[str], body_start: int) -> list[Utterance]:
    utterances: list[Utterance] = []
    pending_timestamp: str | None = None
    last_timestamp: str | None = None

    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped or _GEMINI_SKIP_RE.match(stripped):
            continue
        if GEMINI_DURATION_RE.search(stripped):
            continue
        if _is_standalone_date_line(stripped) or _is_likely_title_line(stripped):
            continue

        if GEMINI_TIMESTAMP_ONLY_RE.match(stripped):
            pending_timestamp = stripped
            last_timestamp = stripped
            continue

        match = GEMINI_LINE_RE.match(stripped)
        if match:
            text = match.group("text").strip()
            if not text:
                continue
            timestamp = match.group("timestamp")
            pending_timestamp = None
            last_timestamp = timestamp
            utterances.append(
                Utterance(
                    timestamp=timestamp,
                    timestamp_seconds=timestamp_to_seconds(timestamp),
                    speaker=normalize_speaker_name(match.group("name")),
                    speaker_org=None,
                    text=text,
                )
            )
            continue

        speaker_match = GEMINI_SPEAKER_LINE_RE.match(stripped)
        if speaker_match and (pending_timestamp or last_timestamp):
            if _is_likely_title_line(stripped):
                continue
            name = speaker_match.group("name").strip()
            if len(name.split()) < 2 and name.isupper():
                continue
            text = speaker_match.group("text").strip()
            if not text:
                continue
            timestamp = pending_timestamp or last_timestamp or "0:00:00"
            pending_timestamp = None
            last_timestamp = timestamp
            utterances.append(
                Utterance(
                    timestamp=timestamp,
                    timestamp_seconds=timestamp_to_seconds(timestamp),
                    speaker=normalize_speaker_name(speaker_match.group("name")),
                    speaker_org=None,
                    text=text,
                )
            )

    return utterances


def _parse_zoom_vtt_body(lines: list[str]) -> list[Utterance]:
    utterances: list[Utterance] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.isdigit():
            idx += 1
            if idx >= len(lines):
                break
            stripped = lines[idx].strip()
        time_match = ZOOM_VTT_TIME_RE.match(stripped)
        if time_match:
            start_ts = time_match.group("start").split(".")[0]
            idx += 1
            text_parts: list[str] = []
            while idx < len(lines):
                next_line = lines[idx].strip()
                if not next_line:
                    idx += 1
                    break
                if next_line.isdigit() or ZOOM_VTT_TIME_RE.match(next_line):
                    break
                text_parts.append(next_line)
                idx += 1
            full = " ".join(text_parts).strip()
            if not full:
                continue
            speaker = "Unknown"
            text = full
            if ":" in full:
                name_part, _, rest = full.partition(":")
                if rest.strip() and len(name_part) < 60:
                    speaker = normalize_speaker_name(name_part)
                    text = rest.strip()
            utterances.append(
                Utterance(
                    timestamp=start_ts,
                    timestamp_seconds=timestamp_to_seconds(start_ts),
                    speaker=speaker,
                    speaker_org=None,
                    text=text,
                )
            )
            continue
        idx += 1
    return utterances


def _parse_otter_body(lines: list[str], body_start: int = 0) -> list[Utterance]:
    utterances: list[Utterance] = []
    current: Utterance | None = None
    pending: list[str] = []

    def _flush() -> None:
        nonlocal current, pending
        if current is None:
            return
        current.text = _clean_utterance_text(" ".join(pending))
        if current.text:
            utterances.append(current)
        current = None
        pending = []

    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_standalone_date_line(stripped) or _is_likely_title_line(stripped):
            continue
        match = OTTER_LINE_RE.match(stripped)
        if match:
            _flush()
            ts = match.group("timestamp")
            current = Utterance(
                timestamp=ts,
                timestamp_seconds=timestamp_to_seconds(ts),
                speaker=normalize_speaker_name(match.group("name")),
                speaker_org=None,
                text="",
            )
        elif current is not None:
            pending.append(stripped)
    _flush()
    return utterances


def _parse_plain_dialogue_body(lines: list[str], body_start: int = 0) -> list[Utterance]:
    utterances: list[Utterance] = []
    seq = 0
    last_speaker: str | None = None

    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if (
            _is_standalone_date_line(stripped)
            or _is_likely_title_line(stripped)
            or GEMINI_DURATION_RE.search(stripped)
            or _GEMINI_SKIP_RE.match(stripped)
            or GEMINI_TIMESTAMP_ONLY_RE.match(stripped)
        ):
            continue
        match = PLAIN_DIALOGUE_RE.match(stripped)
        if not match:
            continue
        if _is_likely_title_line(stripped):
            continue
        text = match.group("text").strip()
        if not text:
            continue
        speaker = normalize_speaker_name(match.group("name"))
        if not speaker or speaker == "Unknown":
            speaker = last_speaker or "Unknown"
        last_speaker = speaker
        seq += 1
        utterances.append(
            Utterance(
                timestamp=f"line_{seq}",
                timestamp_seconds=seq,
                speaker=speaker,
                speaker_org=None,
                text=text,
            )
        )
    return utterances


def _parse_by_format(
    raw: str,
    fmt: TranscriptFormat,
    universal_meta: tuple[str | None, str | None, int | None, str],
) -> tuple[
    str | None,
    str | None,
    int | None,
    list[Utterance],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    lines = raw.splitlines()
    action_items: list[dict[str, Any]] = []
    screen_shares: list[dict[str, Any]] = []

    if fmt == TranscriptFormat.FATHOM:
        title, duration, body_start = _parse_fathom_header(lines)
        utterances, action_items, screen_shares = _parse_fathom_body(lines, body_start)
        meeting_date = parse_meeting_date(title or "")
        merged = _merge_metadata(universal_meta, (title, meeting_date, duration))
        return (*merged[:3], utterances, action_items, screen_shares, merged[3])

    if fmt == TranscriptFormat.GEMINI:
        title, meeting_date, duration, body_start = _parse_gemini_metadata(lines)
        utterances = _parse_gemini_body(lines, body_start)
        if not meeting_date and title:
            meeting_date = parse_meeting_date(title)
        merged = _merge_metadata(universal_meta, (title, meeting_date, duration))
        return (*merged[:3], utterances, action_items, screen_shares, merged[3])

    if fmt == TranscriptFormat.ZOOM_VTT:
        title = lines[0].strip() if lines and lines[0].strip() else None
        if title and title.upper() == "WEBVTT":
            title = None
        utterances = _parse_zoom_vtt_body(lines)
        merged = _merge_metadata(
            universal_meta,
            (title, parse_meeting_date(title or ""), None),
        )
        return (*merged[:3], utterances, action_items, screen_shares, merged[3])

    if fmt == TranscriptFormat.OTTER:
        body_start = _find_dialogue_body_start(lines)
        title = universal_meta[0]
        if not title:
            for ln in lines[:body_start]:
                if _is_likely_title_line(ln.strip()):
                    title = ln.strip()
                    break
        utterances = _parse_otter_body(lines, body_start)
        merged = _merge_metadata(
            universal_meta,
            (title, parse_meeting_date(title or ""), None),
        )
        return (*merged[:3], utterances, action_items, screen_shares, merged[3])

    if fmt == TranscriptFormat.PLAIN_DIALOGUE:
        body_start = _find_dialogue_body_start(lines)
        title = universal_meta[0]
        meeting_date = universal_meta[1]
        duration = universal_meta[2]
        utterances = _parse_plain_dialogue_body(lines, body_start)
        merged = _merge_metadata(universal_meta, (title, meeting_date, duration))
        return (*merged[:3], utterances, action_items, screen_shares, merged[3])

    # UNKNOWN: try each parser in order of likelihood, return first with utterances
    for candidate in (
        TranscriptFormat.GEMINI,
        TranscriptFormat.FATHOM,
        TranscriptFormat.PLAIN_DIALOGUE,
        TranscriptFormat.OTTER,
        TranscriptFormat.ZOOM_VTT,
    ):
        t, md, dur, u, ai, ss, src = _parse_by_format(raw, candidate, universal_meta)
        if u:
            return t, md, dur, u, ai, ss, src

    u_title, u_date, u_dur, u_src = universal_meta
    return u_title, u_date, u_dur, [], action_items, screen_shares, u_src


def parse_transcript_text(
    raw: str,
    transcript_id: str,
    source_file: str,
    format_hint: TranscriptFormat | None = None,
) -> ParsedTranscript:
    """Parse raw transcript text into a :class:`ParsedTranscript`."""
    prepared, normalization_applied = prepare_text_for_parsing(raw)
    lines = prepared.splitlines()
    universal_meta = _extract_universal_metadata(lines)

    fmt = format_hint or detect_format(prepared)
    title, meeting_date, duration, utterances, action_items, screen_shares, metadata_source = (
        _parse_by_format(prepared, fmt, universal_meta)
    )

    # If detection picked a format but parsing failed, retry as unknown
    if not utterances and fmt != TranscriptFormat.UNKNOWN:
        title, meeting_date, duration, utterances, action_items, screen_shares, metadata_source = (
            _parse_by_format(prepared, TranscriptFormat.UNKNOWN, universal_meta)
        )
        if utterances:
            fmt = TranscriptFormat.UNKNOWN

    participants = unique_preserve_order(u.speaker for u in utterances)

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
        detected_format=fmt.value,
        normalization_applied=normalization_applied,
        metadata_source=metadata_source,
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

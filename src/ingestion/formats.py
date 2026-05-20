"""Transcript format detection and metadata.

Scans raw text to guess which export format was used (Fathom, Gemini, Zoom
VTT, Otter, plain dialogue) so the parser can dispatch to the right logic.
"""

from __future__ import annotations

import re
from enum import Enum


class TranscriptFormat(str, Enum):
    FATHOM = "fathom"
    GEMINI = "gemini"
    ZOOM_VTT = "zoom_vtt"
    OTTER = "otter"
    PLAIN_DIALOGUE = "plain_dialogue"
    UNKNOWN = "unknown"


# Order matters: more specific patterns before generic ones.
_GEMINI_STRONG_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}\s+[A-Za-z][^:]+:\s*\S")
_GEMINI_TIMESTAMP_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_GEMINI_FOOTER_RE = re.compile(
    r"Transcription ended after\s+\d{1,2}:\d{2}:\d{2}",
    re.IGNORECASE,
)
_GEMINI_DATE_LINE_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+)?\d{1,2},?\s*\d{4}$"
)
_GEMINI_TITLE_RE = re.compile(
    r".*(transcript|meeting|call|standup|sync|daily).*( - |:)",
    re.IGNORECASE,
)

_DETECTION_ORDER: list[tuple[TranscriptFormat, re.Pattern[str]]] = [
    (
        TranscriptFormat.ZOOM_VTT,
        re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->"),
    ),
    (
        TranscriptFormat.GEMINI,
        _GEMINI_STRONG_RE,
    ),
    (
        TranscriptFormat.FATHOM,
        re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*-\s*[A-Za-z]"),
    ),
    (
        TranscriptFormat.OTTER,
        re.compile(r"^[A-Za-z][A-Za-z\s]{1,40}\s+\d{1,2}:\d{2}(?::\d{2})?\s*$"),
    ),
    (
        TranscriptFormat.PLAIN_DIALOGUE,
        re.compile(r"^[A-Za-z][A-Za-z\s]{1,40}:\s+\S"),
    ),
]

_FORMAT_META: dict[TranscriptFormat, dict[str, str]] = {
    TranscriptFormat.FATHOM: {
        "label": "Fathom",
        "description": "Timestamped speaker lines with optional org in parentheses.",
        "example": "0:09 - Speaker Name (org)",
    },
    TranscriptFormat.GEMINI: {
        "label": "Gemini / Google Meet",
        "description": "HH:MM:SS timestamp, speaker name, and text on one line.",
        "example": "00:00:00 Speaker Name: text...",
    },
    TranscriptFormat.ZOOM_VTT: {
        "label": "Zoom VTT",
        "description": "WebVTT-style cue blocks with millisecond timestamps.",
        "example": "00:00:05.120 --> 00:00:08.440",
    },
    TranscriptFormat.OTTER: {
        "label": "Otter.ai",
        "description": "Speaker name followed by a timestamp on the same line.",
        "example": "Speaker Name  0:00",
    },
    TranscriptFormat.PLAIN_DIALOGUE: {
        "label": "Plain dialogue",
        "description": "Speaker name and text separated by a colon, no timestamp.",
        "example": "Speaker Name: text...",
    },
    TranscriptFormat.UNKNOWN: {
        "label": "Unknown",
        "description": "Format could not be detected; best-effort parsing is used.",
        "example": "",
    },
}


def format_label(fmt: TranscriptFormat) -> str:
    return _FORMAT_META.get(fmt, _FORMAT_META[TranscriptFormat.UNKNOWN])["label"]


def format_metadata(fmt: TranscriptFormat) -> dict[str, str]:
    return dict(_FORMAT_META.get(fmt, _FORMAT_META[TranscriptFormat.UNKNOWN]))


def _detect_gemini_signals(text: str) -> int:
    """Score Gemini/Google Meet signals (higher = more likely Gemini)."""
    if not text.strip():
        return 0

    score = 0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    ts_only = sum(1 for ln in lines if _GEMINI_TIMESTAMP_ONLY_RE.match(ln))
    if ts_only >= 3:
        score += 3
    elif ts_only >= 1:
        score += 1

    if any(_GEMINI_FOOTER_RE.search(ln) for ln in lines):
        score += 3

    header = lines[:15]
    if any(_GEMINI_DATE_LINE_RE.match(ln) for ln in header):
        score += 2

    if any(_GEMINI_TITLE_RE.match(ln) for ln in header):
        score += 2

    strong = sum(1 for ln in lines if _GEMINI_STRONG_RE.match(ln))
    if strong >= 5:
        score += 4
    elif strong >= 2:
        score += 2
    elif strong >= 1:
        score += 1

    # Split-line Gemini: timestamp line followed by Speaker: line
    split_pairs = 0
    for idx, ln in enumerate(lines[:-1]):
        if _GEMINI_TIMESTAMP_ONLY_RE.match(ln):
            nxt = lines[idx + 1]
            if re.match(r"^[A-Za-z][^\n:]+:\s*\S", nxt):
                split_pairs += 1
    if split_pairs >= 2:
        score += 3

    return score


def detect_format(text: str, scan_lines: int = 100) -> TranscriptFormat:
    """Guess transcript format by scanning the first non-empty lines."""
    if not text or not text.strip():
        return TranscriptFormat.UNKNOWN

    # Strong signal: many HH:MM:SS Speaker: lines anywhere (Gemini / Meet exports).
    gemini_total = sum(
        1 for line in text.splitlines() if _GEMINI_STRONG_RE.match(line.strip())
    )
    if gemini_total >= 5:
        return TranscriptFormat.GEMINI

    # Signal-based Gemini detection (split-line timestamps, footer, date header).
    gemini_score = _detect_gemini_signals(text)
    if gemini_score >= 5:
        return TranscriptFormat.GEMINI

    counts: dict[TranscriptFormat, int] = {fmt: 0 for fmt, _ in _DETECTION_ORDER}
    examined = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        examined += 1
        if examined > scan_lines:
            break
        for fmt, pattern in _DETECTION_ORDER:
            if pattern.match(stripped):
                counts[fmt] += 1

    # Pick the format with the most matching lines; tie-break by detection order.
    best_fmt = TranscriptFormat.UNKNOWN
    best_count = 0
    for fmt, _ in _DETECTION_ORDER:
        if counts[fmt] > best_count:
            best_count = counts[fmt]
            best_fmt = fmt

    if best_count >= 2:
        # If plain dialogue wins but Gemini signals are strong, prefer Gemini.
        if best_fmt == TranscriptFormat.PLAIN_DIALOGUE and gemini_score >= 4:
            return TranscriptFormat.GEMINI
        return best_fmt
    if best_count == 1:
        if best_fmt == TranscriptFormat.PLAIN_DIALOGUE and gemini_score >= 4:
            return TranscriptFormat.GEMINI
        return best_fmt

    if gemini_score >= 3:
        return TranscriptFormat.GEMINI

    return TranscriptFormat.UNKNOWN


def detect_format_with_confidence(text: str) -> tuple[TranscriptFormat, str]:
    """Return format and confidence level (high, medium, low)."""
    fmt = detect_format(text)
    if fmt == TranscriptFormat.UNKNOWN:
        return fmt, "low"

    counts: dict[TranscriptFormat, int] = {f: 0 for f in TranscriptFormat if f != TranscriptFormat.UNKNOWN}
    for line in text.splitlines()[:100]:
        stripped = line.strip()
        if not stripped:
            continue
        for f, pattern in _DETECTION_ORDER:
            if pattern.match(stripped):
                counts[f] += 1

    top = counts.get(fmt, 0)
    total = sum(counts.values())

    # Boost confidence for Gemini when signal score is high
    if fmt == TranscriptFormat.GEMINI:
        signal = _detect_gemini_signals(text)
        if signal >= 6:
            return fmt, "high"
        if signal >= 4:
            return fmt, "medium"

    if top >= 5 and top >= total * 0.6:
        return fmt, "high"
    if top >= 2:
        return fmt, "medium"
    return fmt, "low"


__all__ = [
    "TranscriptFormat",
    "detect_format",
    "detect_format_with_confidence",
    "format_label",
    "format_metadata",
]

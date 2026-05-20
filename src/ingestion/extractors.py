"""Extract plain text from uploaded transcript files (.txt, .docx)."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

# Gemini docx exports often put the timestamp on its own line:
#   00:00:00
#   Speaker Name: text...
_GEMINI_SPLIT_TS_RE = re.compile(
    r"(?m)^(\d{2}:\d{2}:\d{2})\s*\n\s*([A-Za-z][^\n:]+:)"
)
# Prefer the spoken-word transcript block, not "Transcript\nSummary" in notes.
_TRANSCRIPT_MARKERS = [
    re.compile(r"📖\s*Transcript", re.IGNORECASE),
    # Section header only — do not match "Meeting - Transcript" inside a title line.
    re.compile(r"^-\s*Transcript\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(
        r"\nTranscript\s*\n(?:[A-Z][a-z]{2,9}\s+\d{1,2},?\s*\d{4}|\d{2}:\d{2}:\d{2})",
        re.IGNORECASE,
    ),
]
_TRANSCRIPT_END_RE = re.compile(
    r"Transcription ended after(?:\s+\d{1,2}:\d{2}:\d{2})?",
    re.IGNORECASE,
)
# Strip common boilerplate lines from pasted/docx exports
_BOILERPLATE_LINE_RE = re.compile(
    r"^(📝|📖|Notes|Meeting records|Attachments|"
    r"Please provide feedback|Get tips|Suggested next steps|"
    r"You should review|This editable transcript|Invited )",
    re.IGNORECASE,
)

SUPPORTED_EXTENSIONS = (".txt", ".docx")


def get_supported_extensions() -> list[str]:
    return list(SUPPORTED_EXTENSIONS)


def extract_from_txt(file_bytes: bytes) -> str:
    """Decode UTF-8 text; fall back to latin-1 then lossy UTF-8."""
    if not file_bytes:
        return ""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            return file_bytes.decode("utf-8", errors="ignore")


def normalize_transcript_text(text: str) -> str:
    """Clean up transcript text before parsing.

    - Prefer the Transcript section when Notes + Transcript are bundled.
    - Join Gemini-style timestamp lines with the following speaker line.
    - Strip common boilerplate header lines.
    """
    if not text:
        return ""

    section_start: int | None = None
    for marker in _TRANSCRIPT_MARKERS:
        match = marker.search(text)
        if match:
            section_start = match.start()
            break
    if section_start is not None:
        text = text[section_start:]
        end_match = _TRANSCRIPT_END_RE.search(text)
        if end_match:
            text = text[: end_match.end()]

    text = _GEMINI_SPLIT_TS_RE.sub(r"\1 \2", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _BOILERPLATE_LINE_RE.match(stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def normalize_extracted_text(text: str) -> str:
    """Backward-compatible alias for :func:`normalize_transcript_text`."""
    return normalize_transcript_text(text)


def prepare_text_for_parsing(text: str) -> tuple[str, bool]:
    """Normalize transcript text for parsing.

    Returns ``(normalized_text, was_modified)``.
    """
    if not text:
        return "", False
    normalized = normalize_transcript_text(text)
    was_modified = normalized != text
    return normalized, was_modified


def extract_from_docx(file_bytes: bytes) -> str:
    """Pull paragraph text from a Word document."""
    from docx import Document

    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return normalize_transcript_text("\n".join(paragraphs))


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract transcript text based on file extension."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        raw = extract_from_txt(file_bytes)
        return normalize_transcript_text(raw)
    if ext == ".docx":
        return extract_from_docx(file_bytes)
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "extract_from_docx",
    "extract_from_txt",
    "extract_text_from_file",
    "get_supported_extensions",
    "normalize_extracted_text",
    "normalize_transcript_text",
    "prepare_text_for_parsing",
]

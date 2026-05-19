"""High-level ingestion service.

The CLI script in ``scripts/ingest_transcripts.py`` and the Streamlit admin
page both go through these functions so the two stay in lock-step. Everything
here is safe to call from the UI: no ``print``, no ``argparse``, every
function returns a structured result.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROCESSED_DIR, RAW_DIR
from src.ingestion.chunker import Chunk, chunk_transcript
from src.ingestion.parser import (
    ParsedTranscript,
    parse_transcript_file,
    parse_transcript_text,
    save_parsed_transcript,
)
from src.retrieval.vector_store import TranscriptVectorStore


@dataclass
class IngestResult:
    """Summary of one ingestion run."""

    transcript_id: str
    meeting_title: str | None
    meeting_date: str | None
    duration_minutes: int | None
    participants: list[str]
    utterance_count: int
    chunk_count: int
    action_item_count: int
    raw_path: str
    json_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "duration_minutes": self.duration_minutes,
            "participants": self.participants,
            "utterance_count": self.utterance_count,
            "chunk_count": self.chunk_count,
            "action_item_count": self.action_item_count,
            "raw_path": self.raw_path,
            "json_path": self.json_path,
        }


@dataclass
class PreviewResult:
    """Lightweight parse-only preview (no embedding, no disk writes)."""

    meeting_title: str | None
    meeting_date: str | None
    duration_minutes: int | None
    participants: list[str]
    utterance_count: int
    action_item_count: int
    screen_share_count: int
    estimated_chunks: int
    sample_utterances: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _slugify(text: str) -> str:
    """Best-effort filename-safe slug."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", text)
    norm = norm.encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_").lower()
    return norm[:40]


def generate_transcript_id(title_hint: str | None = None) -> str:
    """Build a unique, human-readable transcript id."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(title_hint or "")
    return f"transcript_{stamp}" + (f"_{slug}" if slug else "")


def list_indexed_transcripts(
    store: TranscriptVectorStore | None = None,
) -> list[dict[str, Any]]:
    """Return aggregated transcript info from the vector store."""
    store = store or TranscriptVectorStore()
    return store.list_transcripts()


def list_raw_files() -> list[dict[str, Any]]:
    """Return one entry per .txt in data/raw/ with file metadata."""
    entries: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        stat = path.stat()
        entries.append(
            {
                "transcript_id": path.stem,
                "filename": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return entries


def list_processed_files() -> set[str]:
    return {p.stem for p in PROCESSED_DIR.glob("*.json")}


def preview_parse(raw_text: str) -> PreviewResult:
    """Parse without writing anything to disk. For the 'Preview' button."""
    parsed = parse_transcript_text(
        raw=raw_text,
        transcript_id="__preview__",
        source_file="__preview__.txt",
    )
    chunks = chunk_transcript(parsed)

    sample = [
        {
            "timestamp": u.timestamp,
            "speaker": u.speaker,
            "text": (u.text[:160] + "...") if len(u.text) > 160 else u.text,
        }
        for u in parsed.utterances[:5]
    ]

    warnings: list[str] = []
    if not parsed.utterances:
        warnings.append(
            "No utterances detected. Make sure the transcript uses the "
            "expected speaker line format: '0:00 - Speaker Name (org)'."
        )
    if not parsed.meeting_title:
        warnings.append("No meeting title detected (the first non-empty line).")
    if parsed.utterances and not parsed.meeting_date:
        warnings.append("Meeting date could not be parsed from the title.")

    return PreviewResult(
        meeting_title=parsed.meeting_title,
        meeting_date=parsed.meeting_date,
        duration_minutes=parsed.duration_minutes,
        participants=parsed.participants,
        utterance_count=len(parsed.utterances),
        action_item_count=len(parsed.action_items),
        screen_share_count=len(parsed.screen_shares),
        estimated_chunks=len(chunks),
        sample_utterances=sample,
        warnings=warnings,
    )


def _persist_inputs(
    parsed: ParsedTranscript, raw_text: str
) -> tuple[Path, Path]:
    raw_path = RAW_DIR / f"{parsed.transcript_id}.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    json_path = PROCESSED_DIR / f"{parsed.transcript_id}.json"
    save_parsed_transcript(parsed, json_path)
    return raw_path, json_path


def _index_chunks(
    chunks: list[Chunk],
    transcript_id: str,
    store: TranscriptVectorStore,
    replace: bool,
) -> None:
    if replace:
        store.delete_transcript(transcript_id)
    store.add_chunks(chunks)


def ingest_single_transcript(
    raw_text: str,
    transcript_id: str | None = None,
    title_hint: str | None = None,
    store: TranscriptVectorStore | None = None,
    replace: bool = True,
) -> IngestResult:
    """Parse, persist, chunk, embed and index a single transcript."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Transcript text is empty.")

    store = store or TranscriptVectorStore()
    tid = transcript_id or generate_transcript_id(title_hint=title_hint)

    parsed = parse_transcript_text(
        raw=raw_text,
        transcript_id=tid,
        source_file=f"{tid}.txt",
    )
    if not parsed.utterances:
        raise ValueError(
            "No utterances were parsed. The text does not look like the "
            "expected speaker-line format ('0:00 - Speaker Name')."
        )

    raw_path, json_path = _persist_inputs(parsed, raw_text)
    chunks = chunk_transcript(parsed)
    _index_chunks(chunks, parsed.transcript_id, store, replace=replace)

    return IngestResult(
        transcript_id=parsed.transcript_id,
        meeting_title=parsed.meeting_title,
        meeting_date=parsed.meeting_date,
        duration_minutes=parsed.duration_minutes,
        participants=parsed.participants,
        utterance_count=len(parsed.utterances),
        chunk_count=len(chunks),
        action_item_count=len(parsed.action_items),
        raw_path=str(raw_path),
        json_path=str(json_path),
    )


def ingest_from_file(
    path: Path,
    transcript_id: str | None = None,
    store: TranscriptVectorStore | None = None,
    copy_into_raw: bool = False,
    replace: bool = True,
) -> IngestResult:
    """Ingest a transcript that already lives somewhere on disk."""
    store = store or TranscriptVectorStore()
    tid = transcript_id or path.stem

    if copy_into_raw and path.parent != RAW_DIR:
        dest = RAW_DIR / f"{tid}.txt"
        shutil.copyfile(path, dest)
        path = dest

    parsed = parse_transcript_file(path, transcript_id=tid)
    if not parsed.utterances:
        raise ValueError(
            f"No utterances were parsed from {path.name}. Check the format."
        )

    json_path = PROCESSED_DIR / f"{parsed.transcript_id}.json"
    save_parsed_transcript(parsed, json_path)

    chunks = chunk_transcript(parsed)
    _index_chunks(chunks, parsed.transcript_id, store, replace=replace)

    return IngestResult(
        transcript_id=parsed.transcript_id,
        meeting_title=parsed.meeting_title,
        meeting_date=parsed.meeting_date,
        duration_minutes=parsed.duration_minutes,
        participants=parsed.participants,
        utterance_count=len(parsed.utterances),
        chunk_count=len(chunks),
        action_item_count=len(parsed.action_items),
        raw_path=str(path),
        json_path=str(json_path),
    )


def delete_transcript_fully(
    transcript_id: str,
    store: TranscriptVectorStore | None = None,
    remove_files: bool = True,
) -> dict[str, Any]:
    """Drop the transcript from the index and (optionally) from disk."""
    store = store or TranscriptVectorStore()
    removed_chunks = store.delete_transcript(transcript_id)

    removed_files: list[str] = []
    if remove_files:
        for candidate in (
            RAW_DIR / f"{transcript_id}.txt",
            PROCESSED_DIR / f"{transcript_id}.json",
        ):
            if candidate.exists():
                candidate.unlink()
                removed_files.append(candidate.name)

    return {
        "transcript_id": transcript_id,
        "chunks_removed": removed_chunks,
        "files_removed": removed_files,
    }


def reindex_all(
    store: TranscriptVectorStore | None = None,
    reset: bool = True,
) -> list[IngestResult]:
    """Re-index every .txt in data/raw/. Optionally wipes the collection."""
    store = store or TranscriptVectorStore()
    if reset:
        store.reset()

    results: list[IngestResult] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        try:
            results.append(
                ingest_from_file(
                    path,
                    transcript_id=path.stem,
                    store=store,
                    copy_into_raw=False,
                    replace=not reset,
                )
            )
        except Exception as exc:
            results.append(
                IngestResult(
                    transcript_id=path.stem,
                    meeting_title=None,
                    meeting_date=None,
                    duration_minutes=None,
                    participants=[],
                    utterance_count=0,
                    chunk_count=0,
                    action_item_count=0,
                    raw_path=str(path),
                    json_path=f"ERROR: {exc}",
                )
            )
    return results


__all__ = [
    "IngestResult",
    "PreviewResult",
    "delete_transcript_fully",
    "generate_transcript_id",
    "ingest_from_file",
    "ingest_single_transcript",
    "list_indexed_transcripts",
    "list_processed_files",
    "list_raw_files",
    "preview_parse",
    "reindex_all",
]

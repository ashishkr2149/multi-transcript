"""Authoritative transcript inventory for grounded answers.

Merges Chroma index metadata with processed JSON on disk so counts, dates,
and titles do not depend on which chunks happen to be retrieved.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.config import PROCESSED_DIR
from src.retrieval.vector_store import TranscriptVectorStore

_DATE_ONLY_TITLE_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+)?\d{1,2},?\s*\d{4}$"
)


@dataclass
class TranscriptRecord:
    transcript_id: str
    meeting_title: str | None
    meeting_date: str | None
    duration_minutes: int | None
    participants: list[str]
    chunk_count: int
    utterance_count: int | None
    detected_format: str | None = None

    @property
    def display_title(self) -> str:
        if self.meeting_title and not _DATE_ONLY_TITLE_RE.match(self.meeting_title.strip()):
            return self.meeting_title
        if self.meeting_date:
            return f"Meeting on {self.meeting_date}"
        return self.transcript_id


class TranscriptCatalog:
    """Ground-truth list of indexed transcripts."""

    def __init__(self, store: TranscriptVectorStore | None = None) -> None:
        self._store = store or TranscriptVectorStore()
        self._records: list[TranscriptRecord] | None = None

    def _load_processed(self, transcript_id: str) -> dict | None:
        path = PROCESSED_DIR / f"{transcript_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def refresh(self) -> list[TranscriptRecord]:
        """Rebuild the catalog from index + processed files."""
        index_rows = self._store.list_transcripts()
        records: list[TranscriptRecord] = []

        for row in index_rows:
            tid = row["transcript_id"]
            processed = self._load_processed(tid) or {}
            participants = processed.get("participants") or sorted(row.get("speakers") or [])
            records.append(
                TranscriptRecord(
                    transcript_id=tid,
                    meeting_title=processed.get("meeting_title") or row.get("meeting_title") or None,
                    meeting_date=processed.get("meeting_date") or row.get("meeting_date") or None,
                    duration_minutes=processed.get("duration_minutes"),
                    participants=list(participants),
                    chunk_count=int(row.get("chunk_count") or 0),
                    utterance_count=len(processed.get("utterances") or []) or None,
                    detected_format=processed.get("detected_format"),
                )
            )

        records.sort(key=lambda r: r.meeting_date or r.transcript_id)
        self._records = records
        return records

    def list_all(self) -> list[TranscriptRecord]:
        if self._records is None:
            return self.refresh()
        return list(self._records)

    def count(self) -> int:
        return len(self.list_all())

    def get(self, transcript_id: str) -> TranscriptRecord | None:
        for rec in self.list_all():
            if rec.transcript_id == transcript_id:
                return rec
        return None

    def unique_dates(self) -> list[str]:
        dates = sorted({r.meeting_date for r in self.list_all() if r.meeting_date})
        return dates

    def format_for_prompt(self, mode: str = "full") -> str:
        """Render numbered catalog for LLM context."""
        records = self.list_all()
        if not records:
            return "(No meetings are indexed in the system.)"

        if mode == "summary":
            return (
                f"Total indexed meetings: {len(records)}. "
                f"Dates: {', '.join(self.unique_dates()) or 'unknown'}."
            )

        lines = [
            "INDEXED MEETINGS (authoritative — use for counts, dates, and meeting identity)",
            "--------------------------------------------------------------------------------",
        ]
        for idx, rec in enumerate(records, start=1):
            title = rec.display_title
            date_part = f" ({rec.meeting_date})" if rec.meeting_date else ""
            duration_part = (
                f", {rec.duration_minutes} min" if rec.duration_minutes else ""
            )
            speaker_part = ""
            if rec.participants and mode == "full":
                speaker_part = f" | speakers: {', '.join(rec.participants[:8])}"
                if len(rec.participants) > 8:
                    speaker_part += ", ..."
            lines.append(
                f"{idx}. {title}{date_part}{duration_part} "
                f"[transcript_id: {rec.transcript_id}]{speaker_part}"
            )
        return "\n".join(lines)

    def inventory_preface(self) -> str:
        """Short factual preface for inventory-style questions."""
        records = self.list_all()
        if not records:
            return "There are 0 indexed meetings in the system."
        dates = self.unique_dates()
        date_text = ", ".join(dates) if dates else "dates unknown"
        return (
            f"There are exactly {len(records)} indexed meetings. "
            f"Meeting dates (ISO): {date_text}. "
            f"Use one section per transcript_id when summarising each meeting."
        )


def title_looks_like_date_only(title: str | None, meeting_date: str | None) -> bool:
    """True when title is only a calendar date (legacy mis-parse)."""
    if not title:
        return False
    if _DATE_ONLY_TITLE_RE.match(title.strip()):
        return True
    if meeting_date and title.strip() == meeting_date:
        return True
    return False

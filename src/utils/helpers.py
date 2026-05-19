"""Misc helpers used across modules."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Iterable

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_meeting_date(title: str, default_year: int | None = None) -> str | None:
    """Best-effort parse of a meeting title like 'Impromptu Zoom Meeting - May 04'.

    Returns ISO-format date string ``YYYY-MM-DD`` or ``None`` if not parseable.
    """
    if not title:
        return None
    text = title.lower()
    match = re.search(r"([a-z]+)\s+(\d{1,2})(?:,?\s+(\d{4}))?", text)
    if not match:
        return None
    month_name, day_str, year_str = match.groups()
    month = _MONTHS.get(month_name)
    if month is None:
        return None
    try:
        day = int(day_str)
    except ValueError:
        return None
    year = int(year_str) if year_str else (default_year or _dt.date.today().year)
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert 'M:SS' or 'H:MM:SS' to integer seconds. Returns 0 on failure."""
    if not timestamp:
        return 0
    parts = timestamp.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    return 0


def normalize_speaker_name(name: str) -> str:
    """Clean a raw speaker name string."""
    if not name:
        return "Unknown"
    return re.sub(r"\s+", " ", name).strip()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Return unique values preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result

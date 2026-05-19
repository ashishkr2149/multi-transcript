"""Shared bootstrap for every Streamlit page.

Adds the project root to ``sys.path`` and applies the global theme so each
page only has to call :func:`page_setup` right after ``st.set_page_config``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.theme import apply_theme  # noqa: E402


def page_setup(title: str = "Multi-Transcript Q&A", icon: str = "\U0001F4DA") -> None:
    """Apply the global theme. Call once per page after ``st.set_page_config``."""
    apply_theme(page_title=title, page_icon=icon)


__all__ = ["PROJECT_ROOT", "page_setup"]

"""Reusable styled UI fragments built on top of the theme.

Every helper here writes HTML through ``st.markdown`` so we can use the CSS
classes defined in :mod:`src.ui.theme`. Keep this file free of business
logic - it should be a pure presentation layer.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import streamlit as st


# A badge spec is either (label, kind) or (label, kind, dot). ``kind`` is one
# of ``""``, ``"primary"``, ``"success"``, ``"warning"`` or ``"danger"``.
BadgeSpec = tuple[str, str] | tuple[str, str, bool]


def _escape(text: str | None) -> str:
    return html.escape(text) if text else ""


def _badge_html(label: str, kind: str = "", dot: bool = False) -> str:
    klass = "mt-badge" + (f" mt-badge-{kind}" if kind else "")
    dot_html = '<span class="mt-badge-dot"></span>' if dot else ""
    return f'<span class="{klass}">{dot_html}{_escape(label)}</span>'


def badge(label: str, kind: str = "", dot: bool = False) -> None:
    st.markdown(_badge_html(label, kind, dot), unsafe_allow_html=True)


def hero(
    title: str,
    subtitle: str,
    badges: Sequence[BadgeSpec] | None = None,
) -> None:
    """Render a quiet hero panel.

    ``badges`` is a list of ``(label, kind)`` or ``(label, kind, dot)``
    tuples. The dot is a small colored circle rendered via CSS; never
    embed raw HTML in ``label``.
    """
    badge_html = ""
    if badges:
        items = []
        for spec in badges:
            if len(spec) == 3:
                label, kind, dot = spec
            else:
                label, kind = spec  # type: ignore[misc]
                dot = False
            items.append(_badge_html(label, kind, dot))
        badge_html = '<div class="mt-hero-badges">' + "".join(items) + "</div>"

    st.markdown(
        f"""
        <div class="mt-hero">
            <h1 class="mt-hero-title">{_escape(title)}</h1>
            <p class="mt-hero-subtitle">{_escape(subtitle)}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def subtitle(text: str) -> None:
    st.markdown(
        f'<p style="color: var(--text-muted); margin: -0.4rem 0 1rem 0; font-size: 0.9rem;">'
        f'{_escape(text)}</p>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    st.markdown(
        f"""
        <div style="margin: 1.2rem 0 0.8rem 0;">
            <h2 style="margin: 0;">{_escape(title)}</h2>
            {f'<p style="color: var(--text-muted); margin: 0.25rem 0 0 0; font-size: 0.88rem;">{_escape(description)}</p>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider_label(text: str) -> None:
    """Subtle uppercase section divider used to break up dense pages."""
    st.markdown(
        f'<div class="mt-subtle-divider">{_escape(text)}</div>',
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="mt-feature-card">
            <div class="mt-feature-icon">{_escape(icon)}</div>
            <h3 class="mt-feature-title">{_escape(title)}</h3>
            <p class="mt-feature-desc">{_escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def transcript_card(
    title: str,
    transcript_id: str,
    date: str | None,
    chunks: int,
    speakers: Iterable[str],
    indexed: bool = True,
    duration_minutes: int | None = None,
) -> None:
    status_badge = (
        _badge_html("Indexed", "success", dot=True)
        if indexed
        else _badge_html("Not indexed", "warning", dot=True)
    )
    speakers_list = ", ".join(speakers) or "-"
    duration_part = (
        f'<div class="mt-stat"><strong>{duration_minutes}</strong>min</div>'
        if duration_minutes
        else ""
    )
    st.markdown(
        f"""
        <div class="mt-card">
            <div style="display:flex; justify-content: space-between; gap: 0.8rem; align-items:flex-start;">
                <div style="flex:1; min-width:0;">
                    <p class="mt-card-title">{_escape(title)}</p>
                    <p class="mt-card-meta">
                        <code>{_escape(transcript_id)}</code>
                        &nbsp;&middot;&nbsp; {_escape(date or "no date")}
                    </p>
                </div>
                <div>{status_badge}</div>
            </div>
            <div class="mt-stat-row">
                <div class="mt-stat"><strong>{chunks}</strong>chunks</div>
                {duration_part}
                <div class="mt-stat" style="max-width: 100%;">
                    <strong>Speakers:</strong> {_escape(speakers_list)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_row(stats: list[tuple[str, str | int]]) -> None:
    columns = st.columns(len(stats))
    for col, (label, value) in zip(columns, stats):
        col.metric(label=label, value=value)


def mini_stat_block(stats: list[tuple[str, str | int]]) -> None:
    """A compact list of label/value pairs - used in the sidebar."""
    rows = "".join(
        f'<div class="mt-mini-stat">'
        f'<span class="mt-mini-stat-label">{_escape(label)}</span>'
        f'<span class="mt-mini-stat-value">{_escape(str(value))}</span>'
        f"</div>"
        for label, value in stats
    )
    st.markdown(
        f'<div style="background: var(--bg-elevated); border: 1px solid var(--border); '
        f'border-radius: 10px; padding: 0.4rem 0.8rem; margin-bottom: 0.8rem;">{rows}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="mt-card" style="text-align:center; padding: 2.2rem 1.5rem;">
            <div style="font-size: 1.8rem; margin-bottom: 0.6rem;">{_escape(icon)}</div>
            <h3 style="margin: 0 0 0.3rem 0;">{_escape(title)}</h3>
            <p style="color: var(--text-muted); margin: 0;">{_escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def brand(title: str = "Multi-Transcript Q&A", subtitle_text: str | None = "Meeting intelligence") -> None:
    """Sidebar brand mark + tagline."""
    initials = "M"
    if title:
        words = title.split()
        if words:
            initials = words[0][:1].upper()
    st.markdown(
        f"""
        <div style="padding: 0.3rem 0.2rem 0.9rem 0.2rem;">
            <div class="mt-brand">
                <span class="mt-brand-mark">{initials}</span>
                <span>{_escape(title)}</span>
            </div>
            {f'<p class="mt-sidebar-subtitle">{_escape(subtitle_text)}</p>' if subtitle_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Keep ``gradient_title`` as a backward-compat shim that now renders plain text.
def gradient_title(text: str) -> None:  # pragma: no cover - deprecated
    st.markdown(f"<h2>{_escape(text)}</h2>", unsafe_allow_html=True)


__all__ = [
    "BadgeSpec",
    "badge",
    "brand",
    "divider_label",
    "empty_state",
    "feature_card",
    "gradient_title",
    "hero",
    "mini_stat_block",
    "section_header",
    "stat_row",
    "subtitle",
    "transcript_card",
]

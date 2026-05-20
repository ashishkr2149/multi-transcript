"""Admin page - manage transcripts (view, add, delete, re-index)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Admin - Multi-Transcript Q&A",
    page_icon="\u2699\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import OPENAI_API_KEY, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.ingestion.extractors import extract_text_from_file  # noqa: E402
from src.ingestion.formats import TranscriptFormat, format_label  # noqa: E402
from src.ingestion.service import (  # noqa: E402
    IngestResult,
    delete_transcript_fully,
    generate_transcript_id,
    ingest_single_transcript,
    list_indexed_transcripts,
    list_processed_files,
    list_raw_files,
    preview_parse,
    reindex_all,
)
from src.retrieval.vector_store import TranscriptVectorStore  # noqa: E402
from src.ui.bootstrap import page_setup  # noqa: E402
from src.ui.components import (  # noqa: E402
    brand,
    divider_label,
    empty_state,
    hero,
    mini_stat_block,
    section_header,
    transcript_card,
)

page_setup()


@st.cache_resource(show_spinner=False)
def get_store() -> TranscriptVectorStore:
    return TranscriptVectorStore()


def _bust_cache() -> None:
    """Make sure the chat page sees the latest store state too."""
    st.cache_resource.clear()


def _format_ingest_result(result: IngestResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Utterances", result.utterance_count)
    cols[1].metric("Chunks", result.chunk_count)
    cols[2].metric("Speakers", len(result.participants))
    cols[3].metric("Action items", result.action_item_count)

    details = []
    if result.meeting_title:
        details.append(f"**Title:** {result.meeting_title}")
    if result.meeting_date:
        details.append(f"**Date:** {result.meeting_date}")
    if result.duration_minutes:
        details.append(f"**Duration:** {result.duration_minutes} min")
    if result.participants:
        details.append(f"**Speakers:** {', '.join(result.participants)}")
    if details:
        st.markdown("  \n".join(details))


def _render_sidebar() -> None:
    store = get_store()
    transcripts = list_indexed_transcripts(store)
    raw_files = list_raw_files()

    with st.sidebar:
        brand("Admin", "Manage transcripts & the index")

        divider_label("System")
        mini_stat_block(
            [
                ("Transcripts", len(transcripts)),
                ("Chunks", store.count()),
                ("Raw files", len(raw_files)),
            ]
        )

        api_state = "Connected" if OPENAI_API_KEY else "Missing key"
        api_kind = "success" if OPENAI_API_KEY else "danger"
        st.markdown(
            f"<span class='mt-badge mt-badge-{api_kind}'>"
            f"<span class='mt-badge-dot'></span>OpenAI: {api_state}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def tab_overview(store: TranscriptVectorStore) -> None:
    transcripts = list_indexed_transcripts(store)
    raw_files = list_raw_files()
    raw_ids = {entry["transcript_id"] for entry in raw_files}
    indexed_ids = {t["transcript_id"] for t in transcripts}
    only_raw = raw_ids - indexed_ids
    only_index = indexed_ids - raw_ids

    section_header(
        "System overview",
        "A snapshot of the transcripts you have ingested and the chunks they produced.",
    )

    cols = st.columns(4)
    cols[0].metric("Indexed transcripts", len(transcripts))
    cols[1].metric("Total chunks", store.count())
    cols[2].metric("Raw files on disk", len(raw_files))
    avg = (store.count() / len(transcripts)) if transcripts else 0
    cols[3].metric("Avg chunks / transcript", f"{avg:.1f}")

    if only_raw:
        st.warning(
            f"**{len(only_raw)} raw file(s) not indexed yet:** "
            + ", ".join(f"`{tid}`" for tid in sorted(only_raw))
            + "\n\nGo to **Manage & Reindex** to bring the index up to date."
        )
    if only_index:
        st.info(
            f"{len(only_index)} transcript(s) are in the index but have no raw file: "
            + ", ".join(f"`{tid}`" for tid in sorted(only_index))
        )

    if not transcripts:
        empty_state(
            icon="\U0001F4DD",
            title="No transcripts yet",
            message="Use the 'Add Transcript' tab to paste or upload your first meeting transcript.",
        )
        return

    st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
    try:
        from src.ingestion.service import list_catalog_records

        catalog_rows = {r["transcript_id"]: r for r in list_catalog_records(store)}
    except Exception:
        catalog_rows = {}

    for t in transcripts:
        tid = t["transcript_id"]
        cat = catalog_rows.get(tid, {})
        title = cat.get("display_title") or t.get("meeting_title") or tid
        if cat.get("title_is_date_only"):
            st.caption(
                f"`{tid}`: title looks like date only — re-index after re-uploading "
                "with the updated parser."
            )
        transcript_card(
            title=title,
            transcript_id=tid,
            date=t.get("meeting_date"),
            chunks=t["chunk_count"],
            speakers=t.get("speakers") or [],
        )


def tab_add(store: TranscriptVectorStore) -> None:
    section_header(
        "Add a new transcript",
        "Paste text or upload a .txt / .docx file. The system auto-detects the format, then parses, chunks, embeds and indexes.",
    )

    with st.expander("Supported transcript formats", expanded=False):
        st.markdown(
            """
            | Format | Example |
            |--------|---------|
            | **Fathom** | `0:09 - Speaker Name (org)` with text on following lines |
            | **Gemini / Google Meet** | `00:00:00 Speaker Name: text on same line` |
            | **Zoom VTT** | `00:00:05.120 --> 00:00:08.440` cue blocks |
            | **Otter.ai** | `Speaker Name  0:00` with text below |
            | **Plain dialogue** | `Speaker Name: text...` (no timestamp) |

            For Word exports, upload the `.docx` and paste only the **Transcript** section if Notes are included.
            """
        )

    if "admin_preview" not in st.session_state:
        st.session_state["admin_preview"] = None
    if "admin_raw_text" not in st.session_state:
        st.session_state["admin_raw_text"] = ""

    mode_col, _ = st.columns([0.5, 0.5])
    input_mode = mode_col.radio(
        "Input method",
        options=["Paste text", "Upload file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_text = ""
    if input_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload a transcript",
            type=["txt", "docx"],
            help="Supported: .txt (plain text) or .docx (Word document, e.g. Gemini notes export).",
        )
        if uploaded is not None:
            try:
                raw_text = extract_text_from_file(
                    uploaded.getvalue(),
                    uploaded.name,
                )
                st.session_state["admin_raw_text"] = raw_text
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                raw_text = st.session_state.get("admin_raw_text", "")
        else:
            raw_text = st.session_state.get("admin_raw_text", "")
        if raw_text:
            with st.expander("Preview file contents", expanded=False):
                st.code(raw_text[:1200] + ("\n..." if len(raw_text) > 1200 else ""))
    else:
        raw_text = st.text_area(
            "Transcript text",
            value=st.session_state.get("admin_raw_text", ""),
            placeholder=(
                "Gemini / Google Meet example:\n"
                "00:00:00 Speaker One: Hello everyone.\n"
                "00:01:30 Speaker Two: Thanks for joining.\n\n"
                "Fathom example:\n"
                "0:09 - Speaker Name (org)\n"
                "  Can you hear me okay?"
            ),
            height=300,
            label_visibility="collapsed",
        )
        st.session_state["admin_raw_text"] = raw_text

    col_id, col_buttons = st.columns([0.6, 0.4])
    default_id = st.session_state.get("admin_transcript_id") or generate_transcript_id()
    transcript_id = col_id.text_input(
        "Transcript ID",
        value=default_id,
        help="A unique identifier. Auto-generated; feel free to edit.",
    )
    st.session_state["admin_transcript_id"] = transcript_id

    with col_buttons:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        preview_clicked = bcol1.button(
            "Preview parse",
            use_container_width=True,
            disabled=not raw_text.strip(),
        )
        ingest_clicked = bcol2.button(
            "Add & index",
            type="primary",
            use_container_width=True,
            disabled=not raw_text.strip(),
        )

    if preview_clicked:
        with st.spinner("Parsing transcript..."):
            try:
                st.session_state["admin_preview"] = preview_parse(raw_text)
            except Exception as exc:  # pragma: no cover - defensive
                st.session_state["admin_preview"] = None
                st.error(f"Preview failed: {exc}")

    if st.session_state.get("admin_preview"):
        preview = st.session_state["admin_preview"]
        st.markdown("---")
        section_header("Preview", "What the parser found, before any embedding work.")

        if preview.detected_format and preview.detected_format != "unknown":
            label = format_label(TranscriptFormat(preview.detected_format))
            conf = preview.format_confidence
            st.markdown(
                f'<span class="mt-badge mt-badge-success">'
                f'<span class="mt-badge-dot"></span>Format: {label} ({conf} confidence)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="mt-badge mt-badge-warning">'
                '<span class="mt-badge-dot"></span>Format: unknown</span>',
                unsafe_allow_html=True,
            )

        cols = st.columns(4)
        cols[0].metric("Utterances", preview.utterance_count)
        cols[1].metric("Speakers", len(preview.participants))
        cols[2].metric("Action items", preview.action_item_count)
        cols[3].metric("Est. chunks", preview.estimated_chunks)

        details = []
        if preview.meeting_title:
            details.append(f"**Title:** {preview.meeting_title}")
        if preview.meeting_date:
            details.append(f"**Date:** {preview.meeting_date}")
        if preview.duration_minutes:
            details.append(f"**Duration:** {preview.duration_minutes} min")
        if preview.participants:
            details.append(f"**Speakers:** {', '.join(preview.participants)}")
        if details:
            st.markdown("  \n".join(details))

        for warning in preview.warnings or []:
            st.warning(warning)

        if preview.sample_utterances:
            with st.expander(f"First {len(preview.sample_utterances)} utterances", expanded=False):
                for u in preview.sample_utterances:
                    st.markdown(
                        f"<div style='margin-bottom: 0.55rem;'>"
                        f"<span class='mt-badge mt-badge-primary'>{u['timestamp']}</span> "
                        f"<strong style='margin-left: 4px;'>{u['speaker']}</strong><br>"
                        f"<span style='color: var(--text-muted); font-size: 0.88rem;'>{u['text']}</span></div>",
                        unsafe_allow_html=True,
                    )

    if ingest_clicked:
        with st.spinner("Parsing, chunking and embedding..."):
            try:
                result = ingest_single_transcript(
                    raw_text=raw_text,
                    transcript_id=transcript_id.strip() or None,
                    store=store,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Indexing failed: {exc}")
                return
        fmt_label = format_label(TranscriptFormat(result.detected_format))
        st.success(
            f"Indexed transcript `{result.transcript_id}` "
            f"({fmt_label}, {result.chunk_count} chunks)."
        )
        _format_ingest_result(result)
        st.session_state["admin_preview"] = None
        st.session_state["admin_raw_text"] = ""
        st.session_state["admin_transcript_id"] = None
        _bust_cache()
        st.toast("Transcript added", icon="\u2705")


def tab_manage(store: TranscriptVectorStore) -> None:
    section_header(
        "Manage & Reindex",
        "Delete transcripts you no longer need, or rebuild the index from the raw files on disk.",
    )

    transcripts = list_indexed_transcripts(store)
    raw_files = list_raw_files()
    raw_ids = {entry["transcript_id"] for entry in raw_files}
    processed_ids = list_processed_files()

    st.markdown("#### Indexed transcripts")
    if not transcripts:
        empty_state(
            icon="\U0001F50D",
            title="Nothing to manage yet",
            message="Add a transcript first from the 'Add Transcript' tab.",
        )
    else:
        for t in transcripts:
            tid = t["transcript_id"]
            cols = st.columns([0.65, 0.15, 0.2])
            with cols[0]:
                transcript_card(
                    title=t.get("meeting_title") or tid,
                    transcript_id=tid,
                    date=t.get("meeting_date"),
                    chunks=t["chunk_count"],
                    speakers=t.get("speakers") or [],
                    indexed=True,
                )
            cols[1].markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
            view_btn = cols[1].button("View raw", key=f"view_{tid}", use_container_width=True)
            cols[2].markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
            delete_btn = cols[2].button(
                "Delete",
                key=f"del_{tid}",
                use_container_width=True,
            )

            if view_btn:
                raw_path = RAW_DIR / f"{tid}.txt"
                if raw_path.exists():
                    with st.expander(f"Raw transcript - {tid}", expanded=True):
                        st.code(raw_path.read_text(encoding="utf-8"), language="markdown")
                else:
                    st.info(f"No raw file on disk for `{tid}` (lives only in the index).")

            if delete_btn:
                st.session_state[f"confirm_delete_{tid}"] = True

            if st.session_state.get(f"confirm_delete_{tid}"):
                with st.container():
                    st.warning(
                        f"This will remove **{t['chunk_count']} chunks** for `{tid}` "
                        "from the index AND delete the raw + parsed files. This cannot be undone."
                    )
                    cc1, cc2, _ = st.columns([0.18, 0.18, 0.64])
                    if cc1.button("Yes, delete", key=f"yes_{tid}", type="primary"):
                        outcome = delete_transcript_fully(tid, store=store)
                        st.session_state[f"confirm_delete_{tid}"] = False
                        _bust_cache()
                        st.success(
                            f"Removed {outcome['chunks_removed']} chunks and "
                            f"{len(outcome['files_removed'])} file(s)."
                        )
                        st.rerun()
                    if cc2.button("Cancel", key=f"no_{tid}"):
                        st.session_state[f"confirm_delete_{tid}"] = False
                        st.rerun()

    st.markdown("---")
    st.markdown("#### Raw files on disk")
    if not raw_files:
        st.info(f"No `.txt` files found in `{RAW_DIR}`.")
    else:
        for entry in raw_files:
            tid = entry["transcript_id"]
            is_indexed = tid in {t["transcript_id"] for t in transcripts}
            badge = (
                '<span class="mt-badge mt-badge-success"><span class="mt-badge-dot"></span>Indexed</span>'
                if is_indexed
                else '<span class="mt-badge mt-badge-warning"><span class="mt-badge-dot"></span>Not indexed</span>'
            )
            has_processed = tid in processed_ids
            processed_badge = (
                '<span class="mt-badge mt-badge-primary">Parsed JSON</span>'
                if has_processed
                else '<span class="mt-badge">No JSON yet</span>'
            )
            st.markdown(
                f"""
                <div class="mt-card">
                    <div style="display:flex; justify-content: space-between; gap: 0.6rem; align-items: flex-start;">
                        <div>
                            <p class="mt-card-title">{entry['filename']}</p>
                            <p class="mt-card-meta">
                                <code>{tid}</code> &middot; {entry['size_kb']} KB &middot; modified {entry['modified']}
                            </p>
                        </div>
                        <div style="white-space: nowrap; display: flex; gap: 6px;">{badge} {processed_badge}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### Re-index everything")
    st.markdown(
        '<p style="color: var(--text-muted);">Rebuild the entire index from the raw files on disk. '
        'Useful after editing files outside the UI or upgrading the chunker.</p>',
        unsafe_allow_html=True,
    )

    rcol1, rcol2 = st.columns([0.5, 0.5])
    if rcol1.button("Re-index (incremental)", use_container_width=True):
        with st.spinner("Re-indexing..."):
            results = []
            for path in sorted(RAW_DIR.glob("*.txt")):
                try:
                    from src.ingestion.service import ingest_from_file
                    results.append(
                        ingest_from_file(path, transcript_id=path.stem, store=store, replace=True)
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"`{path.stem}` failed: {exc}")
        _bust_cache()
        st.success(f"Re-indexed {len(results)} transcript(s).")
        st.rerun()

    if rcol2.button("Full reset & re-index", type="primary", use_container_width=True):
        with st.spinner("Wiping collection and rebuilding..."):
            results = reindex_all(store=store, reset=True)
        _bust_cache()
        ok = [r for r in results if r.chunk_count > 0]
        st.success(
            f"Wiped and rebuilt index. {len(ok)}/{len(results)} transcript(s) indexed."
        )
        st.rerun()


def tab_settings(store: TranscriptVectorStore) -> None:
    section_header(
        "Settings & system info",
        "Useful pointers and a danger-zone for wiping state.",
    )

    cfg_rows = [
        ("Project root", str(PROJECT_ROOT)),
        ("Raw dir", str(RAW_DIR)),
        ("Processed dir", str(PROCESSED_DIR)),
        ("Chunks indexed", str(store.count())),
        ("Indexed transcripts", str(len(list_indexed_transcripts(store)))),
        ("Generated at", datetime.now().isoformat(timespec="seconds")),
    ]
    rows_html = "".join(
        f"""
        <div style="display:flex; justify-content: space-between; gap: 1rem;
                    padding: 0.55rem 0.9rem;
                    border-bottom: 1px solid var(--border);">
            <span style="color: var(--text-muted); font-size: 0.85rem;">{label}</span>
            <code style="background: transparent; border: none; color: var(--text); font-size: 0.82rem;">{value}</code>
        </div>
        """
        for label, value in cfg_rows
    )
    st.markdown(
        f'<div style="background: var(--bg-elevated); border: 1px solid var(--border); '
        f'border-radius: 10px; overflow: hidden;">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    divider_label("Danger zone")
    st.markdown(
        '<p style="color: var(--text-muted); font-size: 0.88rem;">'
        'Drops the entire Chroma collection. '
        'Raw files stay on disk; you can rebuild the index from them with "Re-index everything".</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mt-danger-zone">', unsafe_allow_html=True)
    if st.button("Wipe vector index"):
        st.session_state["confirm_wipe"] = True
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("confirm_wipe"):
        st.error("Are you sure? This removes every chunk from the index.")
        cc1, cc2, _ = st.columns([0.2, 0.2, 0.6])
        if cc1.button("Yes, wipe", key="wipe_yes", type="primary"):
            store.reset()
            _bust_cache()
            st.session_state["confirm_wipe"] = False
            st.success("Vector index wiped. Use Re-index to rebuild.")
            st.rerun()
        if cc2.button("Cancel", key="wipe_no"):
            st.session_state["confirm_wipe"] = False
            st.rerun()


def main() -> None:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Add it to your .env and restart.")
        st.stop()

    store = get_store()
    _render_sidebar()

    hero(
        title="Transcript Admin",
        subtitle=(
            "View, add, delete and re-index your meeting transcripts. "
            "Every change is reflected in the chat immediately."
        ),
        badges=[
            (f"{len(list_indexed_transcripts(store))} transcripts", "primary"),
            (f"{store.count()} chunks indexed", ""),
        ],
    )

    overview, add, manage, settings = st.tabs(
        ["Overview", "Add Transcript", "Manage & Reindex", "Settings"]
    )
    with overview:
        tab_overview(store)
    with add:
        tab_add(store)
    with manage:
        tab_manage(store)
    with settings:
        tab_settings(store)


if __name__ == "__main__":
    main()

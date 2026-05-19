"""Chat page - cross-transcript Q&A with multi-session memory."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Chat - Multi-Transcript Q&A",
    page_icon="\U0001F4AC",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.chat import session as chat_db  # noqa: E402
from src.chat.memory import build_retrieval_query, recent_history  # noqa: E402
from src.config import OPENAI_API_KEY, TOP_K_RESULTS  # noqa: E402
from src.generation.generator import AnswerGenerator  # noqa: E402
from src.generation.prompts import wants_speaker_attribution  # noqa: E402
from src.retrieval.vector_store import TranscriptVectorStore  # noqa: E402
from src.ui.bootstrap import page_setup  # noqa: E402
from src.ui.components import brand, empty_state, hero  # noqa: E402

page_setup()


@st.cache_resource(show_spinner=False)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()


@st.cache_resource(show_spinner=False)
def get_store() -> TranscriptVectorStore:
    return TranscriptVectorStore()


def _ensure_session_state() -> None:
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("show_sources", True)


def _select_or_create_session() -> str:
    sessions = chat_db.list_sessions()
    if not sessions:
        new_session = chat_db.create_session("New chat")
        st.session_state["active_session_id"] = new_session.id
        return new_session.id
    if st.session_state.get("active_session_id") not in {s.id for s in sessions}:
        st.session_state["active_session_id"] = sessions[0].id
    return st.session_state["active_session_id"]


def _render_sidebar(active_session_id: str) -> None:
    with st.sidebar:
        brand("Multi-Transcript Q&A", "Conversational meeting intelligence")

        st.markdown(
            '<div class="mt-subtle-divider" style="margin-top: 0.2rem;">Conversations</div>',
            unsafe_allow_html=True,
        )

        if st.button("New chat", use_container_width=True, type="primary"):
            new_session = chat_db.create_session("New chat")
            st.session_state["active_session_id"] = new_session.id
            st.rerun()

        sessions = chat_db.list_sessions()
        st.markdown(
            '<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True
        )
        for sess in sessions:
            cols = st.columns([0.82, 0.18])
            label = sess.title or "Untitled chat"
            is_active = sess.id == active_session_id
            display_label = (label[:34] + "...") if len(label) > 34 else label
            prefix = "> " if is_active else "  "
            if cols[0].button(
                f"{prefix}{display_label}",
                key=f"open_{sess.id}",
                use_container_width=True,
            ):
                st.session_state["active_session_id"] = sess.id
                st.rerun()
            if cols[1].button("x", key=f"del_{sess.id}", help="Delete chat"):
                chat_db.delete_session(sess.id)
                if st.session_state.get("active_session_id") == sess.id:
                    st.session_state["active_session_id"] = None
                st.rerun()

        st.markdown(
            '<div class="mt-subtle-divider">Display</div>',
            unsafe_allow_html=True,
        )
        st.toggle(
            "Show sources",
            key="show_sources",
            help="Display the meeting excerpts used to generate each answer.",
        )

        with st.expander("Indexed transcripts", expanded=False):
            store = get_store()
            transcripts = store.list_transcripts()
            if not transcripts:
                st.write("No transcripts yet. Use the Admin page to add one.")
            else:
                for t in transcripts:
                    title = t.get("meeting_title") or t["transcript_id"]
                    date = t.get("meeting_date") or "no date"
                    st.markdown(
                        f"<div style='font-size: 0.84rem; margin-bottom: 0.55rem;'>"
                        f"<strong style='color: var(--text);'>{title}</strong><br>"
                        f"<span style='color: var(--text-dim);'>"
                        f"{date} &middot; {t['chunk_count']} chunks</span></div>",
                        unsafe_allow_html=True,
                    )
            st.caption(f"Total chunks: {store.count()}")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)} excerpts)"):
        unique_meetings = sorted(
            {(s.get("meeting_title") or "", s.get("meeting_date") or "") for s in sources}
        )
        if len(unique_meetings) > 1:
            badges = " ".join(
                f"<span class='mt-badge mt-badge-primary'>{title or 'unknown'}</span>"
                for title, _ in unique_meetings
            )
            st.markdown(
                f"<div style='margin-bottom: 0.7rem; display:flex; align-items:center; gap: 0.4rem; flex-wrap: wrap;'>"
                f"<span style='color: var(--text-dim); font-size: 0.82rem;'>Spans</span>"
                f"{badges}</div>",
                unsafe_allow_html=True,
            )
        for src in sources:
            sim = src.get("similarity")
            sim_text = f"similarity {sim:.2f}" if isinstance(sim, (int, float)) else ""
            speakers = ", ".join(src.get("speakers") or []) or "-"
            meeting_title = src.get("meeting_title") or src.get("transcript_id") or ""
            meeting_date = src.get("meeting_date") or "no date"
            time_range = src.get("time_range", "") or ""
            meta_sep = " &middot; " if time_range else ""
            st.markdown(
                f"""
                <div class="mt-source-card">
                    <div class="mt-source-head">
                        <div class="mt-source-title">{meeting_title}</div>
                        <div class="mt-source-meta">{meeting_date}{meta_sep}{time_range}</div>
                    </div>
                    <div class="mt-source-speakers">Speakers: {speakers}{f' &middot; {sim_text}' if sim_text else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(src.get("text") or "", language="markdown")


def _render_messages(messages) -> None:
    for msg in messages:
        avatar = "\U0001F464" if msg.role == "user" else "\U0001F4AB"
        with st.chat_message(msg.role, avatar=avatar):
            st.markdown(msg.content)
            if msg.role == "assistant":
                meta_chunks = []
                if msg.mode:
                    mode_label = "Speaker attribution" if msg.mode == "attribution" else "Synthesised"
                    badge_kind = "primary" if msg.mode == "attribution" else ""
                    klass = "mt-badge" + (f" mt-badge-{badge_kind}" if badge_kind else "")
                    meta_chunks.append(f'<span class="{klass}">{mode_label}</span>')
                if msg.sources:
                    meta_chunks.append(
                        f'<span class="mt-badge">{len(msg.sources)} excerpts</span>'
                    )
                    sources_used = sorted({
                        (s.get("meeting_title") or s.get("transcript_id") or "")
                        for s in msg.sources
                    })
                    if len(sources_used) > 1:
                        meta_chunks.append(
                            f'<span class="mt-badge mt-badge-primary">'
                            f'Across {len(sources_used)} meetings</span>'
                        )
                if meta_chunks:
                    st.markdown(
                        "<div style='margin-top: 0.5rem; display:flex; gap: 6px; flex-wrap: wrap;'>"
                        + " ".join(meta_chunks)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                if st.session_state.get("show_sources"):
                    _render_sources(msg.sources or [])


def _handle_user_question(active_session_id: str, question: str) -> None:
    prior_history = chat_db.get_messages(active_session_id)

    if not any(m.role == "user" for m in prior_history):
        chat_db.rename_session(active_session_id, chat_db.auto_title_from(question))

    chat_db.add_message(active_session_id, role="user", content=question)
    full_history = chat_db.get_messages(active_session_id)

    enriched_query = build_retrieval_query(question, prior_history)
    history_payload = recent_history(full_history)

    generator = get_generator()
    with st.spinner("Searching transcripts and composing answer..."):
        answer = generator.answer(
            question=enriched_query,
            chat_history=history_payload,
            top_k=TOP_K_RESULTS,
            force_attribution=wants_speaker_attribution(question),
        )

    sources_payload = [
        {
            "chunk_id": s.chunk_id,
            "transcript_id": s.transcript_id,
            "meeting_title": s.meeting_title,
            "meeting_date": s.meeting_date,
            "time_range": s.time_range,
            "speakers": s.speakers,
            "primary_speaker": s.primary_speaker,
            "similarity": s.similarity,
            "text": s.text,
        }
        for s in answer.sources
    ]
    chat_db.add_message(
        active_session_id,
        role="assistant",
        content=answer.text,
        sources=sources_payload,
        mode=answer.mode,
    )


def main() -> None:
    _ensure_session_state()

    if not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY is not set. Create a `.env` file at the project "
            "root with `OPENAI_API_KEY=<your key>` and restart the app."
        )
        st.stop()

    active_session_id = _select_or_create_session()
    _render_sidebar(active_session_id)

    session = chat_db.get_session(active_session_id)
    hero(
        title="Conversational Meeting Intelligence",
        subtitle=(
            "Ask anything about your indexed transcripts. Answers synthesise "
            "across meetings automatically; ask 'who...?' to switch to speaker "
            "attribution mode."
        ),
        badges=[
            (f"Active chat: {session.title}" if session else "New chat", "primary"),
            ("Cross-transcript retrieval", ""),
            ("Context-aware follow-ups", ""),
        ],
    )

    store = get_store()
    if store.count() == 0:
        empty_state(
            icon="\U0001F4E5",
            title="No transcripts indexed yet",
            message=(
                "Head to the Admin page to add your first transcript. "
                "You can paste text directly or upload a .txt file."
            ),
        )
        if st.button("Go to Admin", type="primary"):
            try:
                st.switch_page("pages/2_Admin.py")
            except Exception:
                st.info("Open the 'Admin' page from the sidebar.")
        st.stop()

    messages = chat_db.get_messages(active_session_id)

    if not messages:
        st.markdown(
            '<div class="mt-subtle-divider">Try a starter prompt</div>',
            unsafe_allow_html=True,
        )
        starter_cols = st.columns(3)
        starters = [
            "What were the main topics discussed across all meetings?",
            "What is the ETL or campaign workflow being designed?",
            "Who is responsible for monitoring failures?",
        ]
        for col, starter in zip(starter_cols, starters):
            if col.button(starter, key=f"starter_{hash(starter)}", use_container_width=True):
                _handle_user_question(active_session_id, starter)
                st.rerun()

    _render_messages(messages)

    question = st.chat_input("Ask about your meeting transcripts...")
    if question:
        _handle_user_question(active_session_id, question)
        st.rerun()


if __name__ == "__main__":
    main()

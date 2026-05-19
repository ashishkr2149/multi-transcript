"""Landing page for the Multi-Transcript Q&A POC.

The actual functionality lives in the multi-page setup:
    * ``pages/1_Chat.py``  - chat interface
    * ``pages/2_Admin.py`` - transcript management

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Multi-Transcript Q&A",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import OPENAI_API_KEY  # noqa: E402
from src.ingestion.service import list_indexed_transcripts  # noqa: E402
from src.retrieval.vector_store import TranscriptVectorStore  # noqa: E402
from src.ui.bootstrap import page_setup  # noqa: E402
from src.ui.components import brand, feature_card, hero  # noqa: E402

page_setup()


@st.cache_resource(show_spinner=False)
def get_store() -> TranscriptVectorStore:
    return TranscriptVectorStore()


def _safe_switch(target: str) -> None:
    try:
        st.switch_page(target)
    except Exception:
        st.info(f"Open the page from the sidebar: `{target}`")


def _render_sidebar() -> None:
    with st.sidebar:
        brand("Multi-Transcript Q&A", "Cross-transcript meeting intelligence")


def main() -> None:
    _render_sidebar()

    if not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY is not set. Create a `.env` file at the project "
            "root with `OPENAI_API_KEY=<your key>` and restart."
        )
        st.stop()

    store = get_store()
    transcripts = list_indexed_transcripts(store)

    hero(
        title="Talk to your meetings.",
        subtitle=(
            "Ingest meeting transcripts once, then ask anything across them. "
            "Answers synthesise across recordings, with optional speaker attribution "
            "when you explicitly ask 'who said it?'."
        ),
        badges=[
            ("Live", "success", True),
            (f"{len(transcripts)} transcripts", "primary"),
            (f"{store.count()} chunks", ""),
        ],
    )

    cta_cols = st.columns([0.22, 0.22, 0.56])
    if cta_cols[0].button(
        "Open chat", type="primary", use_container_width=True
    ):
        _safe_switch("pages/1_Chat.py")
    if cta_cols[1].button(
        "Open admin", use_container_width=True
    ):
        _safe_switch("pages/2_Admin.py")

    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)

    feature_cols = st.columns(3)
    with feature_cols[0]:
        feature_card(
            icon="\U0001F50D",
            title="Cross-transcript retrieval",
            description=(
                "Every chunk from every transcript lives in one vector index. "
                "Ask a question once - the answer pulls from any meeting that's relevant."
            ),
        )
    with feature_cols[1]:
        feature_card(
            icon="\U0001F464",
            title="Speaker attribution on demand",
            description=(
                "By default, answers stay neutral. Ask 'who said', 'who suggested' or "
                "'who is responsible' and the system switches to attribution mode."
            ),
        )
    with feature_cols[2]:
        feature_card(
            icon="\U0001F4AC",
            title="Context-aware follow-ups",
            description=(
                "Each chat has its own memory. Ask 'What is the ETL process?' then "
                "'Who explained it?' - the second question resolves against the first."
            ),
        )

    st.markdown("<div style='height: 1.4rem;'></div>", unsafe_allow_html=True)

    flow_cols = st.columns(3)
    with flow_cols[0]:
        feature_card(
            icon="\U0001F4E5",
            title="1. Add transcripts",
            description=(
                "Upload a .txt file or paste text directly. The parser handles "
                "speakers, timestamps and action items automatically."
            ),
        )
    with flow_cols[1]:
        feature_card(
            icon="\U0001F9E0",
            title="2. The system indexes",
            description=(
                "Speaker-aware chunking, OpenAI embeddings and a ChromaDB vector "
                "store keep retrieval fast and accurate."
            ),
        )
    with flow_cols[2]:
        feature_card(
            icon="\U0001F4AB",
            title="3. Chat across meetings",
            description=(
                "Open the chat, ask a question, and synthesise insights from "
                "every meeting in your knowledge base."
            ),
        )

    if not transcripts:
        st.markdown(
            """
            <div class="mt-card" style="margin-top: 1.8rem; text-align: center;">
                <h3 style="margin: 0 0 0.4rem 0;">Ready to begin?</h3>
                <p style="color: var(--text-muted); margin-bottom: 1rem;">
                    Head to the Admin page and add your first transcript.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Add a transcript now", type="primary"):
            _safe_switch("pages/2_Admin.py")


if __name__ == "__main__":
    main()

# Multi-Transcript Q&A POC

A polished, multi-page Streamlit application that lets you chat with a
collection of meeting transcripts using Retrieval-Augmented Generation
(RAG). Answers can span multiple meetings; if a topic was discussed across
three different calls, the answer pulls from all three at once.

It supports:

- Cross-transcript correlation (a single ChromaDB collection holds chunks
  from every transcript).
- Speaker attribution **only when asked** (the model stays neutral by
  default and switches to "who said what" mode when you ask "who...?",
  "which speaker...?", "who is responsible...?", etc.).
- Multiple independent chats with their own memory.
- Context-aware follow-ups within a chat (e.g. ask "What is the ETL
  process?" then "Who explained it?" - the second question is resolved
  against the first).
- A modern, glassmorphic dark UI with smooth animations.
- An **Admin page** for paste/upload, deletion and re-indexing - no CLI
  required.

## 1. Prerequisites

- Python 3.10 or later (tested up to 3.14)
- An OpenAI API key

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

Create a `.env` file at the project root (this repo ships with one
populated for the POC):

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=12
MAX_CHUNKS_PER_TRANSCRIPT=3
PER_MEETING_CHUNKS=2
RETRIEVAL_OVERSAMPLE=24
MAX_CHAT_HISTORY=10
```

## 3. Launch the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).
You will see a multi-page app with three entries in the sidebar:

| Page | Purpose |
| --- | --- |
| **App** (`app.py`) | Landing / hub with overview and shortcuts |
| **Chat** (`pages/1_Chat.py`) | Ask questions across all indexed transcripts |
| **Admin** (`pages/2_Admin.py`) | Manage transcripts (add, view, delete, re-index) |

## 4. Add your first transcript

You can add transcripts two ways:

### Option A - From the UI (recommended)

1. Open the app and switch to **Admin**.
2. Use the **Add Transcript** tab.
3. Paste the transcript text (or upload a `.txt` file).
4. Click **Preview parse** to verify what the system detected.
5. Click **Add & index** to parse, chunk, embed and index it.

The chat page will see the new transcript immediately (the caches are
invalidated automatically).

### Option B - From the CLI

Drop `.txt` files into [`data/raw/`](data/raw) then run:

```bash
python -m scripts.ingest_transcripts            # incremental
python -m scripts.ingest_transcripts --force    # re-index everything
python -m scripts.ingest_transcripts --reset    # wipe collection first
```

The four sample transcripts shipped with this repo are already indexed.

### Expected transcript format

The parser recognises the Fathom-style export format:

```
Impromptu Zoom Meeting - May 04
VIEW RECORDING - 24 mins (No highlights):

---

0:09 - Andrew Gaikwad (kainyx.com)
  Can you hear me okay?

0:10 - Learner Park Media
  Yes, I can hear you.
```

## 5. Chat page

- **+ New chat** creates an isolated conversation with its own memory.
- **Sidebar list** lets you switch and delete chats.
- **Show sources** toggles a panel under each assistant message that lists
  the meeting excerpts used to generate the answer.
- Each assistant turn is tagged with the mode the model used:
  - **Synthesised** - default neutral answer.
  - **Speaker attribution** - triggered when you ask "who...?".
- **Quick-start prompts** appear when a chat is empty; click any to start.

## 6. Admin page

The admin page has four tabs:

| Tab | What you can do |
| --- | --- |
| **Overview** | High-level metrics, drift between raw files and the index |
| **Add Transcript** | Paste text or upload a .txt, preview parse output, then index |
| **Manage & Reindex** | View raw, delete a transcript (chunks + files), incremental or full re-index |
| **Settings** | System paths, danger-zone "wipe index" button |

Adding, deleting or re-indexing clears the cached store immediately so the
Chat page sees the updated data on the next interaction.

## 7. How asking questions works

Default (no name-drop):

> **You:** What is the ETL process?
>
> **Assistant:** The ETL process consists of three stages... (synthesised
> from multiple meetings.)

Attribution mode (auto-triggered by keywords like "who", "which speaker",
"responsible"):

> **You:** Who explained the ETL process?
>
> **Assistant:** Andrew Gaikwad explained it across the May 04 and April 29
> meetings... (cited per excerpt.)

The attribution detector lives in
[`src/generation/prompts.py`](src/generation/prompts.py). The pre-retrieval
query is enriched with the latest chat turns by
[`src/chat/memory.py`](src/chat/memory.py) so short follow-ups like "who
explained it?" actually find ETL-related chunks.

## 8. Accuracy and meeting counts

Answers combine an **indexed meeting catalog** (authoritative counts and dates)
with **retrieved excerpts** (what was said). Inventory and per-meeting summary
questions use dedicated retrieval strategies so the model does not confuse
multiple excerpts from one meeting with separate meetings.

After parser updates, re-index from **Admin → Manage & Reindex** so Chroma
metadata (titles, dates) stays in sync.

## 9. Tests

Lightweight unit tests cover parsing, routing, and validation (no network):

```bash
PYTHONPATH=. python tests/test_accuracy.py
PYTHONPATH=. python tests/test_parser_formats.py
PYTHONPATH=. python tests/test_parser.py
PYTHONPATH=. python tests/test_retrieval.py
```

## 10. Useful scripts

- `python -m scripts.ingest_transcripts` - parse + chunk + embed + index.
- `python -m scripts.reset_db` - drop the ChromaDB collection.

## 11. Layout

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design notes,
data flow and component responsibilities.

```
Multi-Transcript-Context/
├── app.py                          # Landing / hub page
├── pages/
│   ├── 1_Chat.py                   # Chat UI
│   └── 2_Admin.py                  # Admin UI (paste/upload/delete/reindex)
├── requirements.txt
├── .env                            # API key + model config (gitignored)
├── data/
│   ├── raw/                        # Original transcripts (.txt)
│   ├── processed/                  # Parsed JSON
│   └── chroma_db/                  # ChromaDB persistent index
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── parser.py               # Raw text -> structured transcript
│   │   ├── chunker.py              # Speaker-aware chunking + token budget
│   │   ├── embedder.py             # OpenAI embeddings
│   │   └── service.py              # Reusable ingest/preview/delete API
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   ├── catalog.py              # Authoritative meeting inventory
│   │   ├── query_router.py         # Intent-based retrieval plans
│   │   └── retriever.py            # Diversified / per-meeting search
│   ├── generation/
│   │   ├── prompts.py              # Grounded standard + attribution prompts
│   │   ├── validator.py            # Post-answer accuracy checks
│   │   └── generator.py            # Catalog + retrieve -> prompt -> generate
│   ├── chat/
│   │   ├── session.py              # SQLite session/message store
│   │   └── memory.py               # History + retrieval-query rewriting
│   ├── ui/
│   │   ├── theme.py                # Global CSS, color palette, animations
│   │   ├── components.py           # Reusable styled UI fragments
│   │   └── bootstrap.py            # Per-page setup (sys.path + theme)
│   └── utils/helpers.py
├── scripts/
│   ├── ingest_transcripts.py
│   └── reset_db.py
└── tests/
    ├── test_accuracy.py
    ├── test_parser_formats.py
    ├── test_parser.py
    └── test_retrieval.py
```

## 12. Design notes for the UI

- Theme lives in [`src/ui/theme.py`](src/ui/theme.py). It injects global
  CSS that controls fonts (Inter + JetBrains Mono), the dark gradient
  background, button hover effects, chat bubbles, expander styling and
  tab styling. Update CSS variables at the top of `_GLOBAL_CSS` to recolor
  everything in one place.
- Reusable UI fragments (`hero`, `feature_card`, `transcript_card`,
  `empty_state`, etc.) live in [`src/ui/components.py`](src/ui/components.py)
  and render through `st.markdown(..., unsafe_allow_html=True)`.
- Every page starts with `st.set_page_config(...)` followed by
  `page_setup()` from [`src/ui/bootstrap.py`](src/ui/bootstrap.py) which
  applies the theme.

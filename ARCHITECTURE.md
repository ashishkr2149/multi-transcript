# Architecture

This document explains how the POC is wired together so a new contributor
can pick it up quickly. Pair it with [`README.md`](README.md) for setup
and usage.

## High-level flow

```
            +---------------------------+
            |     data/raw/*.txt        |   raw Fathom-style transcripts
            +-------------+-------------+
                          |
                          v
            +---------------------------+
            |   src.ingestion.parser    |   speaker / timestamp / action-item
            +-------------+-------------+   extraction -> ParsedTranscript
                          |
                          v
            +---------------------------+
            |   data/processed/*.json   |   structured copy on disk
            +-------------+-------------+
                          |
                          v
            +---------------------------+
            |   src.ingestion.chunker   |   speaker-aware, token-bounded
            +-------------+-------------+   chunks with rich metadata
                          |
                          v
            +---------------------------+
            |  src.ingestion.embedder   |   OpenAI text-embedding-3-small
            +-------------+-------------+
                          |
                          v
            +---------------------------+
            | src.retrieval.vector_store|   ChromaDB (single collection,
            |        (ChromaDB)         |   cosine similarity)
            +-------------+-------------+
                          ^
                          |  retrieval at query-time
                          |
            +---------------------------+
            |  src.retrieval.retriever  |   semantic search, grouping by
            +-------------+-------------+   transcript, context formatting
                          |
                          v
            +---------------------------+
            | src.generation.generator  |   prompt assembly + OpenAI
            +-------------+-------------+   gpt-4o-mini chat completion
                          |
                          v
            +---------------------------+
            |  app.py + pages/*.py      |   Streamlit multi-page UI
            |  src.ui.theme / components|   (landing, chat, admin) with
            +---------------------------+   shared dark glassmorphic theme
                          |
                          v
            +---------------------------+
            |     src.chat.session      |   SQLite chats + messages
            +---------------------------+
```

Admin actions (paste/upload/delete/reindex) go through
`src.ingestion.service` which is the same code path used by the CLI in
`scripts.ingest_transcripts`. After any mutation, the Streamlit
`@st.cache_resource` caches are cleared so the chat page sees the change
on the next interaction.

## Why one ChromaDB collection?

Cross-transcript correlation is the headline feature. The simplest way to
get it - and the way this POC implements it - is to put chunks from every
transcript into a single Chroma collection. A semantic query then naturally
returns the most relevant chunks regardless of source. The
`transcript_id` is stored in metadata so we can still attribute and group.

## Multi-page Streamlit layout

The UI is split into three pages so the chat experience and the admin
experience can evolve independently while sharing a single backend:

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Landing / hub page with feature cards and CTA |
| [`pages/1_Chat.py`](pages/1_Chat.py) | The chat UI (was the original `app.py`) |
| [`pages/2_Admin.py`](pages/2_Admin.py) | Transcript management (add/view/delete/reindex) |

A small shared helper in [`src/ui/bootstrap.py`](src/ui/bootstrap.py)
prepends the project root to `sys.path` and calls
[`src.ui.theme.apply_theme`](src/ui/theme.py) so every page starts from the
same visual baseline. Heavyweight resources (vector store, generator) are
wrapped with `@st.cache_resource` so they're built once per process and
shared across pages.

## Ingestion service (`src/ingestion/service.py`)

The admin page and the CLI both go through this service:

| Function | What it does |
| --- | --- |
| `preview_parse(raw_text)` | Parse without writing anywhere; powers the "Preview parse" button |
| `ingest_single_transcript(raw_text, ...)` | Parse + save raw/JSON + chunk + embed + index |
| `ingest_from_file(path, ...)` | Same, starting from a file already on disk |
| `delete_transcript_fully(transcript_id, ...)` | Drop chunks from the index and remove raw/JSON files |
| `reindex_all(reset=True)` | Wipe and rebuild the index from `data/raw/` |
| `generate_transcript_id(title_hint=...)` | Build a unique, human-readable id |
| `list_indexed_transcripts() / list_raw_files()` | Aggregations used by the admin tabs |

Each ingestion function returns an `IngestResult` so the UI can render
real metrics ("18 chunks indexed, 24 utterances, 2 speakers") instead of
parsing log lines.

## Parsing (`src/ingestion/parser.py`)

The parser walks the transcript line by line. It recognises:

- A header line for the meeting title (everything before `---` that isn't
  the `VIEW RECORDING` line).
- A duration line of the form `VIEW RECORDING - 24 mins`.
- Speaker lines that match
  `^(\d+:\d+(:\d+)?)\s*-\s*<name>(\s*\(<org>\))?$`.
- Indented continuation lines (added to the previous speaker's utterance).
- Inline `ACTION ITEM:` and `SCREEN SHARING:` markers - pulled into their
  own lists but also flagged on the surrounding utterance.

`ParsedTranscript.to_dict()` is what we serialise to
`data/processed/<id>.json`.

## Chunking (`src/ingestion/chunker.py`)

The strategy is **speaker-preserving, token-bounded** chunking:

1. Walk utterances in order.
2. Accumulate them into the current chunk until adding the next one would
   exceed the `CHUNK_SIZE` token budget (default 500, measured with
   `tiktoken cl100k_base`).
3. When the budget is hit, emit the chunk and start a new one carrying
   `overlap_utterances` (default 1) from the tail of the previous chunk so
   context bridges chunk boundaries.
4. Single utterances larger than the budget are kept whole - we never
   split mid-utterance because that would orphan the speaker label.

Each chunk stores:

- The chunk text with inline speaker / timestamp labels
  (`[Andrew Gaikwad @ 1:13]: ...`).
- A separate `text_for_embedding` that includes a header
  (`Meeting: ... Date: ... Speakers: ...`) so the embedding picks up
  topical hints in addition to raw words.
- Metadata: `transcript_id`, `meeting_title`, `meeting_date`,
  `time_range`, `speakers_in_chunk`, `primary_speaker`,
  `has_action_item`, `has_screen_share`, `token_count`.

## Embedding & vector store (`src/ingestion/embedder.py`, `src/retrieval/vector_store.py`)

- Embeddings come from `text-embedding-3-small` (cheap, 1536 dims, plenty
  of quality for this POC).
- ChromaDB runs in **persistent client** mode under `data/chroma_db/`.
- The collection is named `transcript_chunks` and uses cosine distance.
- `add_chunks` runs the embed step in batches of 64 and writes ids,
  embeddings, documents (the readable chunk text) and metadata in one go.

## Retrieval (`src/retrieval/retriever.py`)

`Retriever.retrieve` embeds the incoming question once and asks Chroma for
the top `TOP_K_RESULTS` chunks (default 12). The result is converted into
`RetrievedChunk` dataclasses for easier consumption downstream.

### Accuracy framework (catalog + router + diversified retrieval)

To avoid wrong meeting counts and incomplete cross-meeting answers, the
chat path uses three extra modules:

| Module | Role |
| --- | --- |
| [`src/retrieval/catalog.py`](src/retrieval/catalog.py) | Authoritative list of indexed meetings (merged from Chroma + `data/processed/*.json`). Injected into every LLM context as **INDEXED MEETINGS**. |
| [`src/retrieval/query_router.py`](src/retrieval/query_router.py) | Heuristic intent routing: `inventory`, `per_meeting_summary`, `cross_meeting_synthesis`, `specific_fact`, `attribution`. |
| [`src/generation/validator.py`](src/generation/validator.py) | Post-answer checks (e.g. claimed meeting count vs catalog). |

`AnswerGenerator.answer` (see [`src/generation/generator.py`](src/generation/generator.py)):

1. Refreshes the catalog.
2. Builds a `QueryPlan` from the user's question (not the enriched retrieval string).
3. Retrieves via `retrieve_for_plan` (diversified or per-transcript).
4. Formats context with catalog block + excerpts.
5. Validates the answer and returns `warnings` for the UI.

`Retriever.format_context` groups results by `transcript_id`, sorts the
groups by meeting date and emits:

```
INDEXED MEETINGS (authoritative — use for counts and dates)
1. DAILY: PLG... (2026-02-09) [transcript_id: ...]

MEETING EXCERPTS (evidence — use for what was said)
=== Meeting: DAILY: PLG... (2026-02-09) | transcript_id: ... ===
[Excerpt 1 | chunk_id: ... | 0:00:00 - 0:05:00 | speakers: ...]
...
```

The LLM is instructed to use INDEXED MEETINGS for counts and EXCERPTS for content.

## Prompting (`src/generation/prompts.py`)

Two system prompts, picked per turn:

- **Standard mode** - the default. The model synthesises across excerpts
  and **does not** name speakers unless the meaning truly demands it.
- **Attribution mode** - the model must attribute every relevant claim to
  the speaker shown in the excerpts and cite the meeting.

The selector is a regex over keywords like `who`, `which speaker`,
`responsible for`, `in charge of`, `by whom`, etc. The chosen mode is
stored on the assistant message so the UI can display it.

## Generation (`src/generation/generator.py`)

`AnswerGenerator.answer` ties everything together:

1. Retrieve top-K chunks for the (possibly enriched) question.
2. Format the context.
3. Format the chat history.
4. Build the messages array via `build_messages`.
5. Call `client.chat.completions.create` with `temperature=0.2`.
6. Return an `Answer` containing the text, the mode, the list of
   `AnswerSource`s and the chunk ids used.

## Chat sessions (`src/chat/session.py`)

SQLite, two tables:

```sql
chat_sessions(id, title, created_at, updated_at)
chat_messages(id, session_id, role, content, sources, mode, created_at)
```

`sources` is a JSON blob of `AnswerSource`-like dicts so we can re-render
"Sources used" when the user re-opens a chat.

`create_session`, `list_sessions`, `get_messages`, `rename_session`,
`delete_session`, `add_message` and `auto_title_from` are the small API
the UI consumes.

## Conversation memory (`src/chat/memory.py`)

Two helpers:

- `recent_history(messages, limit)` - the last `MAX_CHAT_HISTORY` messages
  formatted as `{"role", "content"}` dicts. Passed to the LLM verbatim.
- `build_retrieval_query(question, history)` - composes a richer string
  for embedding that includes the last few user turns and a short tail of
  the latest assistant answer. This is what makes follow-ups like "who
  explained it?" work: by themselves they would retrieve unrelated
  chunks, but enriched they find ETL-related chunks again.

The retriever sees the enriched query, while the LLM sees the original
question plus chat history.

## Streamlit UI (`app.py`)

- Sidebar: list of chats, "+ New chat", per-chat delete, sources toggle,
  indexed-transcripts expander.
- Main area: existing messages rendered with `st.chat_message`, then a
  `st.chat_input` at the bottom.
- On submit:
  1. If this is the first user message, auto-name the chat from the
     question.
  2. Save the user turn.
  3. Build the enriched retrieval query.
  4. Call `AnswerGenerator.answer`.
  5. Save the assistant turn with serialised sources + mode.
  6. Rerun to repaint.
- Per-assistant-message expander shows the source excerpts (similarity,
  speakers, meeting and the original chunk text).

## Theme and components (`src/ui/`)

The UI is intentionally factored into three small files:

| File | Role |
| --- | --- |
| [`src/ui/theme.py`](src/ui/theme.py) | All global CSS; `apply_theme()` injects it via `st.markdown`. Edit CSS variables at the top to retint the app. |
| [`src/ui/components.py`](src/ui/components.py) | Reusable HTML fragments: `hero`, `feature_card`, `transcript_card`, `empty_state`, `section_header`, `subtitle`, `stat_row`. |
| [`src/ui/bootstrap.py`](src/ui/bootstrap.py) | `page_setup()` - sys.path bootstrap + theme application; called once per page. |

The aesthetic is a dark glassmorphic look with a violet -> cyan gradient
accent. Buttons, expanders, tabs and chat bubbles get hover transitions
and entrance animations via the CSS in `theme.py`. Streamlit's default
chrome (top-right menu, "Made with Streamlit" footer) is hidden.

## Configuration knobs

All in [`src/config.py`](src/config.py) (overridable via `.env`):

| Knob | Default | Effect |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Reserved (overlap is measured in utterances, see `chunker.py`) |
| `TOP_K_RESULTS` | `12` | Chunks retrieved per question |
| `MAX_CHUNKS_PER_TRANSCRIPT` | `3` | Cap per meeting in diversified retrieval |
| `PER_MEETING_CHUNKS` | `2` | Chunks fetched per meeting for per-meeting summaries |
| `RETRIEVAL_OVERSAMPLE` | `24` | Initial pool size before per-transcript dedupe |
| `INVENTORY_TOP_K` | `0` | Chunks retrieved for pure inventory questions |
| `MAX_CHAT_HISTORY` | `10` | Turns sent to the LLM as history |

### Re-index after parser changes

If transcript parsing logic changes, rebuild the index so Chroma metadata
matches the new titles/dates:

1. Admin → **Manage & Reindex**, or
2. `python -m scripts.ingest_transcripts --reset`

Raw files in `data/raw/` are re-parsed with `normalize_transcript_text()` applied.

## Extending the POC

- **New transcript formats** - add a new parser variant in
  `src/ingestion/parser.py` and let it produce a `ParsedTranscript`. The
  rest of the pipeline is format-agnostic.
- **Filtering** - `TranscriptVectorStore.query` accepts a Chroma `where`
  clause. Surface filters (date range, speaker, transcript id) in the UI
  by passing `where={...}` to `AnswerGenerator.answer`.
- **Re-ranking** - swap the top-K result list through Cohere Rerank or
  similar before formatting context.
- **Hybrid search** - layer a keyword search on top of the embedding
  search for exact-match terms (names, acronyms).

## Limitations to be aware of

- The parser is tuned for Fathom-style exports. Other formats will need a
  new parser branch.
- We do not stream LLM tokens; answers appear in one block. Streaming is a
  drop-in upgrade if/when needed.
- The same OpenAI key is used for embeddings and generation; cost is low
  for this POC scale but you should monitor it.
- ChromaDB persists locally. There is no multi-user concurrency story -
  Streamlit runs single-process which is fine for a POC.

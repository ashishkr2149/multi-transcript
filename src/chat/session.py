"""SQLite-backed chat session store.

Schema:
    chat_sessions(id, title, created_at, updated_at)
    chat_messages(id, session_id, role, content, sources, mode, created_at)

Sessions are independent: each chat keeps its own running history and the
``Retriever`` is called fresh per turn so memory does not leak across chats.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

from src.config import CHAT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    mode TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, id);
"""


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class ChatMessage:
    id: int
    session_id: str
    role: str
    content: str
    sources: list[dict[str, Any]]
    mode: str | None
    created_at: str


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHAT_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def create_session(title: str = "New chat") -> ChatSession:
    init_db()
    session_id = uuid.uuid4().hex[:12]
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, title, ts, ts),
        )
    return ChatSession(id=session_id, title=title, created_at=ts, updated_at=ts)


def list_sessions() -> list[ChatSession]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [ChatSession(**dict(row)) for row in rows]


def get_session(session_id: str) -> ChatSession | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    return ChatSession(**dict(row)) if row else None


def rename_session(session_id: str, title: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
            (title, _now(), session_id),
        )


def delete_session(session_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    mode: str | None = None,
) -> ChatMessage:
    init_db()
    ts = _now()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, sources, mode, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, role, content, sources_json, mode, ts),
        )
        msg_id = cur.lastrowid
        conn.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE id=?",
            (ts, session_id),
        )
    return ChatMessage(
        id=msg_id,
        session_id=session_id,
        role=role,
        content=content,
        sources=sources or [],
        mode=mode,
        created_at=ts,
    )


def get_messages(session_id: str) -> list[ChatMessage]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, session_id, role, content, sources, mode, created_at "
            "FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    out: list[ChatMessage] = []
    for row in rows:
        data = dict(row)
        try:
            data["sources"] = json.loads(data.get("sources") or "[]")
        except json.JSONDecodeError:
            data["sources"] = []
        out.append(ChatMessage(**data))
    return out


def auto_title_from(question: str, max_len: int = 60) -> str:
    """Generate a short title from the user's first question."""
    cleaned = " ".join((question or "").strip().split())
    if not cleaned:
        return "New chat"
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "..."
    return cleaned

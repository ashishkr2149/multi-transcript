"""Conversation memory and retrieval-query rewriting.

The retriever embeds the user's question to find chunks. Short follow-up
questions like "who explained it?" miss relevant context unless we expand
them with what came before. ``build_retrieval_query`` joins the latest user
turns into a richer query while ``recent_history`` keeps the prompt within
a sane token budget.
"""

from __future__ import annotations

from src.chat.session import ChatMessage
from src.config import MAX_CHAT_HISTORY


def recent_history(messages: list[ChatMessage], limit: int = MAX_CHAT_HISTORY) -> list[dict[str, str]]:
    """Return the last ``limit`` messages as ``{"role", "content"}`` dicts."""
    if not messages:
        return []
    tail = messages[-limit:]
    return [{"role": m.role, "content": m.content} for m in tail]


def build_retrieval_query(
    question: str,
    history: list[ChatMessage],
    window: int = 3,
) -> str:
    """Compose an enriched query for the retriever.

    Includes:
    * the last few user turns from this session (excluding the current one),
    * a short tail of the latest assistant answer.

    This makes follow-up questions like "who explained it?" or "tell me more"
    actually retrieve material related to the previously discussed topic.
    """
    parts: list[str] = []

    prior_user_turns = [m.content for m in history if m.role == "user"]
    if prior_user_turns:
        tail = prior_user_turns[-window:]
        parts.extend(f"Previous question: {t}" for t in tail)

    last_assistant = next(
        (m for m in reversed(history) if m.role == "assistant" and m.content), None
    )
    if last_assistant:
        snippet = " ".join(last_assistant.content.split())[:600]
        parts.append(f"Previous answer summary: {snippet}")

    parts.append(f"Current question: {question}")
    return "\n".join(parts)

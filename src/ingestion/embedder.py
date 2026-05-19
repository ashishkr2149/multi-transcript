"""Embedding helper. Wraps the OpenAI embeddings endpoint."""

from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, assert_api_key

_BATCH_SIZE = 64


def _client() -> OpenAI:
    assert_api_key()
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI. Returns one vector per input."""
    if not texts:
        return []

    client = _client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    vectors = embed_texts([query])
    return vectors[0] if vectors else []


def stream_embeddings(texts: Iterable[str]) -> list[list[float]]:
    """Convenience wrapper that accepts any iterable."""
    return embed_texts(list(texts))

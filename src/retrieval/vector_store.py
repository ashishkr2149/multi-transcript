"""ChromaDB-backed vector store for transcript chunks."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import embed_query, embed_texts


class TranscriptVectorStore:
    """Thin wrapper around a ChromaDB collection.

    All chunks from every transcript live in a single collection. Cross-
    transcript retrieval is therefore the default behaviour - the only thing
    that decides whether a chunk surfaces is its semantic similarity to the
    user's question.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def list_transcripts(self) -> list[dict[str, Any]]:
        """Return one entry per indexed transcript with chunk counts."""
        result = self._collection.get(include=["metadatas"])
        agg: dict[str, dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            tid = meta.get("transcript_id")
            if not tid:
                continue
            entry = agg.setdefault(
                tid,
                {
                    "transcript_id": tid,
                    "meeting_title": meta.get("meeting_title", ""),
                    "meeting_date": meta.get("meeting_date", ""),
                    "chunk_count": 0,
                    "speakers": set(),
                },
            )
            entry["chunk_count"] += 1
            for spk in (meta.get("speakers_in_chunk") or "").split(","):
                spk = spk.strip()
                if spk:
                    entry["speakers"].add(spk)
        for entry in agg.values():
            entry["speakers"] = sorted(entry["speakers"])
        return sorted(agg.values(), key=lambda e: e.get("meeting_date") or "")

    def delete_transcript(self, transcript_id: str) -> int:
        """Delete all chunks belonging to a transcript. Returns count removed."""
        result = self._collection.get(where={"transcript_id": transcript_id})
        ids = result.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            self.delete_chunk(chunk.chunk_id)

        embeddings = embed_texts([c.text_for_embedding for c in chunks])
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def delete_chunk(self, chunk_id: str) -> None:
        try:
            self._collection.delete(ids=[chunk_id])
        except Exception:
            pass

    def query(
        self,
        question: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k chunks (across ALL transcripts) for a query."""
        if self.count() == 0:
            return []
        query_embedding = embed_query(question)
        if not query_embedding:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[dict[str, Any]] = []
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 - float(dist) if dist is not None else None
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": similarity,
                }
            )
        return out

    def reset(self) -> None:
        """Drop the collection entirely. Useful for re-indexing from scratch."""
        try:
            self._client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

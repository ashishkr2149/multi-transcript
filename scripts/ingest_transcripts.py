"""End-to-end ingestion script.

Reads every ``*.txt`` file in ``data/raw/``, parses it into a structured
JSON document written to ``data/processed/`` and indexes its chunks into the
ChromaDB collection used by the chat application.

Usage:
    python -m scripts.ingest_transcripts             # incremental: skip indexed
    python -m scripts.ingest_transcripts --force     # re-index everything
    python -m scripts.ingest_transcripts --reset     # wipe collection first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR  # noqa: E402
from src.ingestion.service import (  # noqa: E402
    ingest_from_file,
    list_indexed_transcripts,
    reindex_all,
)
from src.retrieval.vector_store import TranscriptVectorStore  # noqa: E402


def ingest(force: bool = False, reset: bool = False) -> int:
    store = TranscriptVectorStore()

    if reset:
        print("Resetting vector collection and re-indexing everything...")
        results = reindex_all(store=store, reset=True)
        for r in results:
            print(f"  - {r.transcript_id}: {r.chunk_count} chunks "
                  f"({r.utterance_count} utterances)")
        print(f"\nDone. Collection holds {store.count()} chunks "
              f"across {len(store.list_transcripts())} transcripts.")
        return sum(r.chunk_count for r in results)

    indexed = {entry["transcript_id"] for entry in list_indexed_transcripts(store)}
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print(f"No transcripts found in {RAW_DIR}.")
        return 0

    total_chunks = 0
    for path in raw_files:
        transcript_id = path.stem
        if not force and transcript_id in indexed:
            print(f"  - {transcript_id}: already indexed (skip). Use --force to re-index.")
            continue

        print(f"Processing {transcript_id} ...")
        result = ingest_from_file(
            path,
            transcript_id=transcript_id,
            store=store,
            copy_into_raw=False,
            replace=True,
        )
        total_chunks += result.chunk_count
        print(f"  parsed: {result.utterance_count} utterances, "
              f"{len(result.participants)} speakers")
        print(f"  indexed: {result.chunk_count} chunks into ChromaDB")

    print(f"\nDone. Total chunks indexed this run: {total_chunks}.")
    print(f"Collection now holds {store.count()} chunks across "
          f"{len(store.list_transcripts())} transcripts.")
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-index transcripts even if already present.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop the collection before indexing.")
    args = parser.parse_args()
    ingest(force=args.force, reset=args.reset)


if __name__ == "__main__":
    main()

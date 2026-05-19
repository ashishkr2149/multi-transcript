"""Central configuration for the Multi-Transcript Q&A POC."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CHROMA_DIR: Path = DATA_DIR / "chroma_db"
CHAT_DB_PATH: Path = DATA_DIR / "chats.sqlite3"

for _path in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    _path.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "12"))

MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))

CHROMA_COLLECTION_NAME: str = "transcript_chunks"


def assert_api_key() -> None:
    """Raise a clear error if the OpenAI API key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file at the project root "
            "with OPENAI_API_KEY=<your key>."
        )

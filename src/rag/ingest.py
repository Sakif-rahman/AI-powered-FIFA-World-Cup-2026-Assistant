"""Ingestion script: load the team knowledge base into ChromaDB.

Run directly::

    python -m src.rag.ingest

It builds one rich document per team from ``data/team_knowledge.json`` and stores
it (with metadata) in the vector store so the Team Agent can retrieve it via RAG.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from config import config
from src.rag.vector_store import get_vector_store
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _team_to_document(team: dict) -> str:
    """Render a single team record into a retrieval-friendly text blob."""
    titles = ", ".join(str(y) for y in team.get("titles_years", [])) or "none"
    players = ", ".join(team.get("key_players", []))
    return (
        f"Team: {team['name']} ({team.get('nickname', '')})\n"
        f"Confederation: {team.get('confederation', 'N/A')}\n"
        f"Group (2026): {team.get('group', 'N/A')}\n"
        f"FIFA Ranking: {team.get('fifa_rank', 'N/A')}\n"
        f"Head Coach: {team.get('coach', 'N/A')}\n"
        f"Captain: {team.get('captain', 'N/A')}\n"
        f"Key Players: {players}\n"
        f"World Cups Won: {team.get('world_cups_won', 0)} (years: {titles})\n"
        f"Overview: {team.get('summary', '')}"
    )


def build_documents() -> Tuple[List[str], List[str], List[dict]]:
    """Return (documents, ids, metadatas) from the knowledge base JSON."""
    with config.team_knowledge_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []

    for team in data.get("teams", []):
        documents.append(_team_to_document(team))
        ids.append(f"team-{team['name'].lower().replace(' ', '-')}")
        metadatas.append(
            {
                "name": team["name"],
                "group": team.get("group", ""),
                "confederation": team.get("confederation", ""),
                "fifa_rank": team.get("fifa_rank", 0),
            }
        )

    return documents, ids, metadatas


def ingest() -> int:
    """Reset the collection and load all team documents. Returns document count."""
    documents, ids, metadatas = build_documents()
    store = get_vector_store()
    store.reset()
    store.add_documents(documents, ids, metadatas)
    count = store.count()
    logger.info("Ingestion complete: %d team documents indexed.", count)
    return count


if __name__ == "__main__":
    total = ingest()
    print(f"Ingested {total} team documents into ChromaDB at '{config.chroma_dir}'.")

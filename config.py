"""Centralized configuration for the FIFA World Cup 2026 Assistant.

All paths, model names, and tunables live here so the rest of the codebase has a
single source of truth. Values can be overridden via environment variables, which
also makes the app friendly to Streamlit Cloud secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Application-wide configuration (immutable)."""

    # --- Paths ---------------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = field(default_factory=lambda: BASE_DIR / Path(_env("CHROMA_DIR", "chroma_db")))

    # --- Data files ----------------------------------------------------------
    history_csv: Path = BASE_DIR / "data" / "world_cup_history.csv"
    matches_csv: Path = BASE_DIR / "data" / "world_cup_matches.csv"
    rankings_csv: Path = BASE_DIR / "data" / "fifa_rankings.csv"
    schedule_csv: Path = BASE_DIR / "data" / "world_cup_schedule.csv"
    team_knowledge_json: Path = BASE_DIR / "data" / "team_knowledge.json"

    # --- LLM -----------------------------------------------------------------
    llm_model: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_temperature: float = 0.3

    # --- RAG / Vector store --------------------------------------------------
    chroma_collection: str = _env("CHROMA_COLLECTION", "wc2026_teams")
    rag_top_k: int = _env_int("RAG_TOP_K", 4)

    # --- App -----------------------------------------------------------------
    app_title: str = "FIFA World Cup 2026 Assistant"
    log_level: str = _env("LOG_LEVEL", "INFO")
    tournament_year: int = 2026
    host_nations: tuple = ("United States", "Canada", "Mexico")

    @property
    def example_prompts(self) -> List[str]:
        return [
            "Tell me about Argentina",
            "Who won the 2022 World Cup?",
            "Compare Brazil and France",
            "Predict the Group J winner",
            "Show England's schedule",
            "Which country has won the most World Cups?",
        ]


# Singleton-style config instance imported throughout the app.
config = Config()

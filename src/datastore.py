"""Structured-data access layer (CSV + team knowledge JSON).

Loads the CSV datasets once (cached) and exposes typed query helpers. The MCP
tools in :mod:`src.mcp.tools` are thin wrappers over these functions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# --- Cached loaders ----------------------------------------------------------
@lru_cache(maxsize=1)
def _history_df() -> pd.DataFrame:
    return pd.read_csv(config.history_csv)


@lru_cache(maxsize=1)
def _matches_df() -> pd.DataFrame:
    return pd.read_csv(config.matches_csv)


@lru_cache(maxsize=1)
def _rankings_df() -> pd.DataFrame:
    return pd.read_csv(config.rankings_csv)


@lru_cache(maxsize=1)
def _schedule_df() -> pd.DataFrame:
    return pd.read_csv(config.schedule_csv)


@lru_cache(maxsize=1)
def _team_knowledge() -> Dict[str, dict]:
    with config.team_knowledge_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {t["name"].lower(): t for t in data.get("teams", [])}


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


# --- Team knowledge ----------------------------------------------------------
def get_team_record(team_name: str) -> Optional[dict]:
    """Return the structured knowledge-base record for a team, or None."""
    teams = _team_knowledge()
    key = _normalize(team_name)
    if key in teams:
        return teams[key]
    # Fuzzy contains match (e.g. "usa" -> "united states" handled via aliases below).
    for name, record in teams.items():
        if key and (key in name or name in key):
            return record
    return None


def list_teams() -> List[str]:
    return sorted(t["name"] for t in _team_knowledge().values())


# --- History -----------------------------------------------------------------
def get_history_by_year(year: int) -> Optional[dict]:
    df = _history_df()
    row = df[df["year"] == int(year)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def all_history() -> List[dict]:
    return _history_df().to_dict(orient="records")


def titles_by_country() -> Dict[str, int]:
    """Return {country: number_of_titles} from the history table."""
    counts = _history_df()["winner"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


# --- Rankings ----------------------------------------------------------------
def get_ranking(team_name: str) -> Optional[dict]:
    df = _rankings_df()
    key = _normalize(team_name)
    match = df[df["team"].str.lower() == key]
    if match.empty:
        match = df[df["team"].str.lower().str.contains(key, na=False)] if key else match
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def top_ranked(n: int = 10) -> List[dict]:
    return _rankings_df().nsmallest(n, "rank").to_dict(orient="records")


# --- Matches -----------------------------------------------------------------
def get_matches_for_team(team_name: str) -> List[dict]:
    df = _matches_df()
    key = _normalize(team_name)
    mask = (df["home_team"].str.lower().str.contains(key, na=False)) | (
        df["away_team"].str.lower().str.contains(key, na=False)
    )
    return df[mask].to_dict(orient="records")


# --- Schedule ----------------------------------------------------------------
def get_schedule_for_team(team_name: str) -> List[dict]:
    df = _schedule_df()
    key = _normalize(team_name)
    mask = (df["home_team"].str.lower().str.contains(key, na=False)) | (
        df["away_team"].str.lower().str.contains(key, na=False)
    )
    return df[mask].sort_values("date").to_dict(orient="records")


def get_schedule_for_group(group: str) -> List[dict]:
    df = _schedule_df()
    g = (group or "").strip().upper().replace("GROUP", "").strip()
    mask = df["group"].astype(str).str.upper() == g
    return df[mask].sort_values("date").to_dict(orient="records")


def teams_in_group(group: str) -> List[str]:
    """Return distinct real team names appearing in a group's fixtures."""
    matches = get_schedule_for_group(group)
    teams = set()
    for m in matches:
        for side in ("home_team", "away_team"):
            name = str(m[side])
            if "winner" not in name.lower() and "runner" not in name.lower():
                teams.add(name)
    return sorted(teams)

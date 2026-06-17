"""Lightweight entity extraction from user queries (teams, years, groups).

Uses simple, deterministic heuristics (no extra LLM call) which keeps routing fast
and cheap. Aliases handle common short forms like "USA" or "Holland".
"""

from __future__ import annotations

import re
from typing import List, Optional

from src import datastore

_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "america": "United States",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "korea": "Korea Republic",
    "south korea": "Korea Republic",
}


def extract_year(text: str) -> Optional[int]:
    """Return the first 4-digit World Cup-plausible year in the text."""
    for match in re.findall(r"\b(19\d{2}|20\d{2})\b", text):
        year = int(match)
        if 1930 <= year <= 2026:
            return year
    return None


def extract_group(text: str) -> Optional[str]:
    """Return a single-letter group reference like 'J' from 'Group J'."""
    match = re.search(r"group\s+([a-l])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_teams(text: str) -> List[str]:
    """Return all known team names mentioned in the text (order preserved)."""
    lowered = text.lower()
    found: List[str] = []

    for alias, canonical in _ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered) and canonical not in found:
            found.append(canonical)

    for team in datastore.list_teams():
        if team.lower() in lowered and team not in found:
            found.append(team)

    return found


def extract_team(text: str) -> Optional[str]:
    teams = extract_teams(text)
    return teams[0] if teams else None

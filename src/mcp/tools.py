"""Tool implementations exposed through MCP.

These are plain, dependency-light functions returning JSON-serializable dicts.
They are the single source of truth for tool behaviour and are used in two ways:

1. In-process by the LangGraph agents (works on Streamlit Cloud, no subprocess).
2. Registered on an MCP server (:mod:`src.mcp.server`) for spec-compliant access
   from external MCP clients.

Each tool returns a dict with an ``ok`` flag plus a payload, so callers can handle
failures gracefully.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src import datastore
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _ok(**payload: Any) -> Dict[str, Any]:
    return {"ok": True, **payload}


def _err(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def get_team_info(team_name: str) -> Dict[str, Any]:
    """Return knowledge-base info (coach, captain, key players, titles) for a team."""
    logger.info("MCP tool get_team_info(%r)", team_name)
    record = datastore.get_team_record(team_name)
    if not record:
        return _err(f"No team information found for '{team_name}'.")
    return _ok(team=record)


def get_worldcup_history(year: int) -> Dict[str, Any]:
    """Return the result of a specific World Cup edition by year."""
    logger.info("MCP tool get_worldcup_history(%r)", year)
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return _err(f"'{year}' is not a valid year.")
    record = datastore.get_history_by_year(year_int)
    if not record:
        return _err(f"No World Cup was held / recorded in {year_int}.")
    return _ok(history=record)


def get_team_ranking(team_name: str) -> Dict[str, Any]:
    """Return the current FIFA ranking entry for a team."""
    logger.info("MCP tool get_team_ranking(%r)", team_name)
    record = datastore.get_ranking(team_name)
    if not record:
        return _err(f"No FIFA ranking found for '{team_name}'.")
    return _ok(ranking=record)


def get_matches(team_name: str) -> Dict[str, Any]:
    """Return historical World Cup matches involving a team."""
    logger.info("MCP tool get_matches(%r)", team_name)
    matches = datastore.get_matches_for_team(team_name)
    if not matches:
        return _err(f"No recorded matches found for '{team_name}'.")
    return _ok(matches=matches, count=len(matches))


def get_schedule(team_name: str = "", group: str = "") -> Dict[str, Any]:
    """Return 2026 fixtures for a team or a group."""
    logger.info("MCP tool get_schedule(team=%r, group=%r)", team_name, group)
    if group:
        fixtures = datastore.get_schedule_for_group(group)
        if not fixtures:
            return _err(f"No fixtures found for group '{group}'.")
        return _ok(fixtures=fixtures, group=group.upper())
    if team_name:
        fixtures = datastore.get_schedule_for_team(team_name)
        if not fixtures:
            return _err(f"No fixtures found for '{team_name}'.")
        return _ok(fixtures=fixtures, team=team_name)
    return _err("Provide either a team_name or a group.")


def get_titles_table() -> Dict[str, Any]:
    """Return a {country: titles} mapping from World Cup history."""
    return _ok(titles=datastore.titles_by_country())


def get_top_rankings(n: int = 10) -> Dict[str, Any]:
    """Return the top-n FIFA-ranked teams."""
    return _ok(rankings=datastore.top_ranked(int(n)))


# Registry used by the MCP server and the agents.
TOOLS: Dict[str, Any] = {
    "get_team_info": get_team_info,
    "get_worldcup_history": get_worldcup_history,
    "get_team_ranking": get_team_ranking,
    "get_matches": get_matches,
    "get_schedule": get_schedule,
    "get_titles_table": get_titles_table,
    "get_top_rankings": get_top_rankings,
}


def list_tools() -> List[str]:
    return list(TOOLS.keys())

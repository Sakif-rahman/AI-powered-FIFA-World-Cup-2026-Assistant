"""Shared LangGraph state definition."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between nodes in the LangGraph workflow."""

    query: str                 # The user's question.
    route: str                 # Chosen route: team | history | prediction | schedule | general.
    response: str              # Final natural-language answer.
    tool_calls: List[str]      # Names of MCP tools invoked (for transparency/logging).
    context: Dict[str, Any]    # Scratch space for retrieved data.

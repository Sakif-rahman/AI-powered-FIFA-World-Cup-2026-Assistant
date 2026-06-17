"""Supervisor agent: analyzes intent and routes to a specialist agent."""

from __future__ import annotations

from groq_chatbot import get_chatbot
from src.agents.extract import extract_group, extract_teams, extract_year
from src.agents.state import AgentState
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"team", "history", "prediction", "schedule", "general"}

_ROUTER_SYSTEM = (
    "You are the routing supervisor for a FIFA World Cup 2026 assistant. "
    "Classify the user's question into exactly ONE of these routes:\n"
    "- team: facts about a national team (coach, captain, players, style, comparisons)\n"
    "- history: past World Cup results, winners, records, statistics\n"
    "- prediction: forecasts of winners, group outcomes, who is likely to win\n"
    "- schedule: 2026 fixtures, when/where a team plays, group fixtures\n"
    "- general: anything else about the World Cup\n"
    "Respond with ONLY the single route word, nothing else."
)

_KEYWORDS = {
    "prediction": ["predict", "likely", "favourite", "favorite", "who will win", "chances", "forecast", "odds"],
    "schedule": ["schedule", "fixture", "when does", "when do", "kick off", "kickoff", "play next", "venue", "stadium", "what time"],
    "history": ["won the", "winner", "champion", "most world cups", "history", "top scorer", "final in", "hosted", "record"],
    "team": ["tell me about", "captain", "coach", "players", "compare", "squad", "manager", "nickname"],
}


def _keyword_route(query: str) -> str:
    q = query.lower()
    if extract_group(query) and any(k in q for k in _KEYWORDS["schedule"] + ["matches", "group"]):
        # "Show Group J matches" -> schedule
        if "predict" not in q and "win" not in q:
            return "schedule"
    for route, keywords in _KEYWORDS.items():
        if any(k in q for k in keywords):
            return route
    if extract_year(query):
        return "history"
    if extract_teams(query):
        return "team"
    return "general"


def classify(query: str) -> str:
    """Classify a query into a route, using the LLM with a keyword fallback."""
    try:
        raw = get_chatbot().chat(query, system_prompt=_ROUTER_SYSTEM, temperature=0.0, max_tokens=8)
        route = raw.strip().lower().split()[0].strip(".,") if raw.strip() else ""
        if route in VALID_ROUTES:
            logger.info("LLM routed %r -> %s", query, route)
            return route
        logger.warning("LLM returned invalid route %r; using keyword fallback.", raw)
    except Exception:  # noqa: BLE001
        logger.exception("LLM routing failed; using keyword fallback.")

    route = _keyword_route(query)
    logger.info("Keyword routed %r -> %s", query, route)
    return route


def supervisor_node(state: AgentState) -> AgentState:
    """Graph node: decide the route for the current query."""
    query = state["query"]
    route = classify(query)
    return {**state, "route": route, "tool_calls": state.get("tool_calls", [])}


def route_selector(state: AgentState) -> str:
    """Conditional-edge function: map state -> next node name."""
    return state.get("route", "general")

"""Prediction Agent: forecasts outcomes via a transparent scoring algorithm.

This is purely an analytical/educational forecast based on FIFA ranking points and
historical World Cup performance. It contains NO gambling, betting odds, or wagering
functionality.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from groq_chatbot import get_chatbot
from src import datastore
from src.agents.extract import extract_group, extract_teams
from src.agents.state import AgentState
from src.mcp import tools as wc_tools
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Weighting of factors in the composite strength score.
_W_RANKING = 0.70   # current form / FIFA ranking points
_W_TITLES = 0.20    # historical pedigree (World Cups won)
_W_RECENT = 0.10    # appearances in recent finals/semis (pedigree proxy)

_MAX_FIFA_POINTS = 2000.0  # normalization ceiling for ranking points

_SYSTEM = (
    "You are a football analyst. Using ONLY the provided computed strength scores, "
    "explain the prediction clearly and briefly. Emphasise this is an analytical "
    "estimate, not betting advice. Do NOT invent numbers."
)


def _recent_pedigree(team: str) -> float:
    """Crude recent-form proxy: count finals/semis appearances in stored matches."""
    matches = datastore.get_matches_for_team(team)
    weighted = 0.0
    for m in matches:
        stage = str(m.get("stage", "")).lower()
        if "final" == stage:
            weighted += 1.0
        elif "semi" in stage:
            weighted += 0.5
        elif "quarter" in stage:
            weighted += 0.25
    return min(weighted / 4.0, 1.0)  # normalize to [0, 1]


def strength_score(team: str) -> Optional[Dict[str, float]]:
    """Compute a composite 0-100 strength score for a team."""
    ranking = datastore.get_ranking(team)
    record = datastore.get_team_record(team)
    if not ranking and not record:
        return None

    points = float(ranking["points"]) if ranking else 1400.0
    ranking_score = min(points / _MAX_FIFA_POINTS, 1.0)

    titles = int(record["world_cups_won"]) if record else 0
    titles_score = min(titles / 5.0, 1.0)  # Brazil's 5 titles = max

    recent_score = _recent_pedigree(team)

    composite = (
        _W_RANKING * ranking_score
        + _W_TITLES * titles_score
        + _W_RECENT * recent_score
    ) * 100.0

    return {
        "team": team,
        "score": round(composite, 1),
        "fifa_points": points,
        "titles": titles,
    }


def rank_teams(teams: List[str]) -> List[Dict[str, float]]:
    scored = [s for t in teams if (s := strength_score(t))]
    return sorted(scored, key=lambda s: s["score"], reverse=True)


def prediction_node(state: AgentState) -> AgentState:
    """Graph node for prediction questions."""
    query = state["query"]
    tool_calls: List[str] = ["get_team_ranking", "get_titles_table"]

    group = extract_group(query)
    if group:
        candidates = datastore.teams_in_group(group)
        scope = f"Group {group}"
    else:
        named = extract_teams(query)
        candidates = named if len(named) >= 2 else datastore.list_teams()
        scope = "the named teams" if len(named) >= 2 else "the tournament"

    ranked = rank_teams(candidates)
    top = ranked[:4] if "top" in query.lower() or not group else ranked

    if not ranked:
        return {**state, "response": "I couldn't find enough data to make a prediction.",
                "tool_calls": state.get("tool_calls", []) + tool_calls}

    table = "\n".join(
        f"{i+1}. {s['team']}: strength {s['score']}/100 "
        f"(FIFA pts {s['fifa_points']:.0f}, {s['titles']} titles)"
        for i, s in enumerate(top)
    )
    context = (
        f"Predictive strength scores for {scope} "
        f"(weights: ranking {_W_RANKING}, titles {_W_TITLES}, recent form {_W_RECENT}):\n{table}"
    )
    prompt = f"{context}\n\nQuestion: {query}\n\nGive the prediction:"
    answer = get_chatbot().chat(prompt, system_prompt=_SYSTEM)

    return {**state, "response": answer, "tool_calls": state.get("tool_calls", []) + tool_calls}

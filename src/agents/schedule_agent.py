"""Schedule Agent: answers 2026 fixture questions from the schedule CSV."""

from __future__ import annotations

from typing import List

from groq_chatbot import get_chatbot
from src.agents.extract import extract_group, extract_team
from src.agents.state import AgentState
from src.mcp import tools as wc_tools
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a FIFA World Cup 2026 schedule assistant. Present fixtures clearly "
    "(date, time, opponent, venue, city). Use ONLY the provided fixtures. "
    "If asked for the 'next' match, identify the earliest upcoming fixture."
)


def _format_fixtures(fixtures: List[dict]) -> str:
    lines = []
    for f in fixtures:
        lines.append(
            f"- {f['date']} {f['time']} | {f['home_team']} vs {f['away_team']} "
            f"| Group {f['group']} | {f['venue']}, {f['city']} | {f['stage']}"
        )
    return "\n".join(lines)


def schedule_node(state: AgentState) -> AgentState:
    """Graph node for schedule questions."""
    query = state["query"]
    tool_calls: List[str] = ["get_schedule"]

    group = extract_group(query)
    team = extract_team(query)

    if group:
        result = wc_tools.get_schedule(group=group)
        scope = f"Group {group}"
    elif team:
        result = wc_tools.get_schedule(team_name=team)
        scope = team
    else:
        return {
            **state,
            "response": "Please tell me which team or group you'd like the schedule for.",
            "tool_calls": state.get("tool_calls", []) + tool_calls,
        }

    if not result.get("ok"):
        return {**state, "response": result.get("error", "No fixtures found."),
                "tool_calls": state.get("tool_calls", []) + tool_calls}

    context = f"Fixtures for {scope}:\n{_format_fixtures(result['fixtures'])}"
    prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = get_chatbot().chat(prompt, system_prompt=_SYSTEM)

    return {**state, "response": answer, "tool_calls": state.get("tool_calls", []) + tool_calls}

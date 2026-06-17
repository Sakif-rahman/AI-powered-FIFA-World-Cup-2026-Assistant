"""History Agent: answers past-tournament questions via CSV lookup (MCP tools)."""

from __future__ import annotations

from typing import List

from groq_chatbot import get_chatbot
from src.agents.extract import extract_year
from src.agents.state import AgentState
from src.mcp import tools as wc_tools
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a FIFA World Cup historian. Answer using ONLY the provided data. "
    "Be precise with years, scores, and names. Keep the answer concise."
)


def history_node(state: AgentState) -> AgentState:
    """Graph node for history questions."""
    query = state["query"]
    tool_calls: List[str] = []
    blocks: List[str] = []

    year = extract_year(query)
    if year:
        result = wc_tools.get_worldcup_history(year)
        tool_calls.append("get_worldcup_history")
        if result.get("ok"):
            h = result["history"]
            blocks.append(
                f"{h['year']} World Cup (host: {h['host']}): "
                f"Winner {h['winner']}, runner-up {h['runner_up']}, "
                f"third {h['third_place']}, fourth {h['fourth_place']}. "
                f"Top scorer: {h['top_scorer']} ({h['top_scorer_goals']} goals). "
                f"Teams: {h['total_teams']}."
            )
        else:
            blocks.append(result.get("error", ""))

    # Always include the titles table — useful for "most titles" style questions.
    titles = wc_tools.get_titles_table()
    tool_calls.append("get_titles_table")
    if titles.get("ok"):
        ranked = sorted(titles["titles"].items(), key=lambda kv: kv[1], reverse=True)
        blocks.append(
            "World Cup titles won: "
            + ", ".join(f"{country} ({n})" for country, n in ranked)
        )

    context = "\n\n".join(blocks) if blocks else "No matching history data found."
    prompt = f"Data:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = get_chatbot().chat(prompt, system_prompt=_SYSTEM)

    return {**state, "response": answer, "tool_calls": state.get("tool_calls", []) + tool_calls}

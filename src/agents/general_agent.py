"""General Agent: handles open-ended World Cup questions with light context."""

from __future__ import annotations

from groq_chatbot import get_chatbot
from src.agents.state import AgentState
from src.mcp import tools as wc_tools
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a friendly, knowledgeable assistant for the FIFA World Cup 2026, "
    "co-hosted by the United States, Canada, and Mexico (48 teams). Answer helpfully "
    "and accurately. If a question is outside football/World Cup scope, gently steer "
    "back to the tournament."
)


def general_node(state: AgentState) -> AgentState:
    """Graph node for general questions."""
    query = state["query"]

    # Provide a little grounding context (top rankings) for relevance.
    top = wc_tools.get_top_rankings(5)
    context = ""
    if top.get("ok"):
        context = "Current FIFA top 5: " + ", ".join(
            f"{r['team']} (#{r['rank']})" for r in top["rankings"]
        )

    prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:" if context else query
    answer = get_chatbot().chat(prompt, system_prompt=_SYSTEM)

    return {**state, "response": answer, "tool_calls": state.get("tool_calls", []) + ["get_top_rankings"]}

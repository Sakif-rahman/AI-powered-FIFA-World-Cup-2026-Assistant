"""Team Agent: answers team questions using RAG + the get_team_info MCP tool."""

from __future__ import annotations

from typing import List

from groq_chatbot import get_chatbot
from src.agents.extract import extract_teams
from src.agents.state import AgentState
from src.mcp import tools as wc_tools
from src.rag.vector_store import get_vector_store
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a knowledgeable FIFA World Cup team analyst. Answer the user's question "
    "using ONLY the provided context. Be concise, accurate, and well-structured. "
    "If comparing teams, give a balanced view. If the context lacks the answer, say so."
)


def _gather_context(query: str, teams: List[str]) -> tuple[str, List[str]]:
    """Build context text from MCP tool calls + RAG retrieval."""
    tool_calls: List[str] = []
    blocks: List[str] = []

    # Structured facts via MCP tool for each named team.
    for team in teams:
        result = wc_tools.get_team_info(team)
        tool_calls.append("get_team_info")
        if result.get("ok"):
            rec = result["team"]
            blocks.append(
                f"[Structured data: {rec['name']}]\n"
                f"Coach: {rec.get('coach')}; Captain: {rec.get('captain')}; "
                f"FIFA rank: {rec.get('fifa_rank')}; Group: {rec.get('group')}; "
                f"Titles: {rec.get('world_cups_won')} {rec.get('titles_years')}; "
                f"Key players: {', '.join(rec.get('key_players', []))}.\n"
                f"Summary: {rec.get('summary')}"
            )

    # Semantic retrieval (RAG) for richer/loosely-matched context.
    for hit in get_vector_store().query(query):
        blocks.append(f"[Knowledge base]\n{hit['document']}")

    return "\n\n".join(blocks) if blocks else "No relevant context found.", tool_calls


def team_node(state: AgentState) -> AgentState:
    """Graph node for team questions."""
    query = state["query"]
    teams = extract_teams(query)
    context, tool_calls = _gather_context(query, teams)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = get_chatbot().chat(prompt, system_prompt=_SYSTEM)

    return {
        **state,
        "response": answer,
        "tool_calls": state.get("tool_calls", []) + tool_calls + ["rag.query"],
    }

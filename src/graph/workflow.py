"""LangGraph multi-agent workflow.

    User
      |
   Supervisor  (intent analysis + routing)
      |
  -------------------------------------------------
  |          |           |           |           |
 Team     History    Prediction   Schedule    General
 Agent     Agent       Agent        Agent       Agent
  -------------------------------------------------
      |
     END  (final response returned via the assistant)

The supervisor classifies intent and a conditional edge dispatches to exactly one
specialist node. Each specialist produces the final ``response`` and the graph ends.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.general_agent import general_node
from src.agents.history_agent import history_node
from src.agents.prediction_agent import prediction_node
from src.agents.schedule_agent import schedule_node
from src.agents.state import AgentState
from src.agents.supervisor import route_selector, supervisor_node
from src.agents.team_agent import team_node
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_graph() -> Any:
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("team", team_node)
    graph.add_node("history", history_node)
    graph.add_node("prediction", prediction_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("general", general_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_selector,
        {
            "team": "team",
            "history": "history",
            "prediction": "prediction",
            "schedule": "schedule",
            "general": "general",
        },
    )

    for node in ("team", "history", "prediction", "schedule", "general"):
        graph.add_edge(node, END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled.")
    return compiled


@lru_cache(maxsize=1)
def get_app() -> Any:
    """Return a cached, compiled workflow app."""
    return build_graph()


def run_query(query: str) -> dict:
    """Run a user query through the workflow and return the result dict."""
    app = get_app()
    try:
        result = app.invoke({"query": query, "tool_calls": []})
        return {
            "response": result.get("response", "Sorry, I couldn't generate a response."),
            "route": result.get("route", "general"),
            "tool_calls": result.get("tool_calls", []),
        }
    except Exception as exc:  # noqa: BLE001 - graceful failure for the UI
        logger.exception("Workflow execution failed")
        return {
            "response": f"Sorry, something went wrong while processing your question: {exc}",
            "route": "error",
            "tool_calls": [],
        }

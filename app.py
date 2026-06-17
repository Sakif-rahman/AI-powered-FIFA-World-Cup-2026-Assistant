"""Streamlit frontend for the FIFA World Cup 2026 Assistant.

Run locally::

    pip install -r requirements.txt
    streamlit run app.py

On first launch the team knowledge base is automatically ingested into ChromaDB if
the vector store is empty, so no manual setup step is required.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from config import config
from src import datastore
from src.rag.ingest import ingest
from src.rag.vector_store import get_vector_store
from src.graph.workflow import run_query
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=config.app_title,
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Preparing knowledge base...")
def ensure_ingested() -> int:
    """Ensure the vector store is populated; ingest on first run."""
    store = get_vector_store()
    if store.count() == 0:
        logger.info("Vector store empty — running ingestion.")
        return ingest()
    return store.count()


@st.cache_data(show_spinner=False)
def load_stats() -> dict:
    """Load lightweight statistics for the sidebar panel."""
    titles = datastore.titles_by_country()
    most = max(titles.items(), key=lambda kv: kv[1]) if titles else ("N/A", 0)
    return {
        "teams": len(datastore.list_teams()),
        "editions": len(datastore.all_history()),
        "most_titles_team": most[0],
        "most_titles_count": most[1],
        "top_ranked": datastore.top_ranked(5),
        "titles": dict(sorted(titles.items(), key=lambda kv: kv[1], reverse=True)),
    }


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[dict] = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def render_sidebar() -> None:
    stats = load_stats()
    with st.sidebar:
        st.title("🏆 World Cup 2026")
        st.caption("Co-hosts: United States · Canada · Mexico")

        st.subheader("📊 Tournament Stats")
        c1, c2 = st.columns(2)
        c1.metric("Teams (KB)", stats["teams"])
        c2.metric("Editions", stats["editions"])
        st.metric(
            "Most Titles",
            f"{stats['most_titles_team']} ({stats['most_titles_count']})",
        )

        st.subheader("🥇 Top FIFA Ranked")
        st.dataframe(
            pd.DataFrame(stats["top_ranked"])[["rank", "team", "points"]],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("🔎 Team Selector")
        team = st.selectbox("Pick a team to ask about", ["—"] + datastore.list_teams())
        if team != "—":
            tc1, tc2 = st.columns(2)
            if tc1.button("ℹ️ Info", use_container_width=True):
                st.session_state.pending_prompt = f"Tell me about {team}"
            if tc2.button("📅 Schedule", use_container_width=True):
                st.session_state.pending_prompt = f"Show {team}'s schedule"

        st.subheader("💡 Example Prompts")
        for i, ex in enumerate(config.example_prompts):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_prompt = ex

        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def handle_query(prompt: str) -> None:
    """Run a query through the workflow and append messages to history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_query(prompt)
        st.markdown(result["response"])
        st.caption(
            f"🧭 Route: `{result['route']}`  ·  🛠️ Tools: "
            f"`{', '.join(dict.fromkeys(result['tool_calls'])) or 'none'}`"
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "route": result["route"],
            "tool_calls": result["tool_calls"],
        }
    )


def main() -> None:
    init_state()
    ensure_ingested()
    render_sidebar()

    st.title("🏆 FIFA World Cup 2026 Assistant")
    st.caption(
        "Ask about teams, history, predictions, and schedules — powered by LangGraph, "
        "MCP tools, RAG (ChromaDB), and Groq."
    )

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("route"):
                tools = ", ".join(dict.fromkeys(msg.get("tool_calls", []))) or "none"
                st.caption(f"🧭 Route: `{msg['route']}`  ·  🛠️ Tools: `{tools}`")

    # Handle a queued prompt from a sidebar button.
    pending = st.session_state.pending_prompt
    if pending:
        st.session_state.pending_prompt = None
        handle_query(pending)

    # Chat input.
    if prompt := st.chat_input("Ask me about the FIFA World Cup 2026..."):
        handle_query(prompt)


if __name__ == "__main__":
    main()

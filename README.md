# 🏆 FIFA World Cup 2026 Assistant

A production-ready, multi-agent AI assistant that answers questions about the FIFA
World Cup 2026 — participating teams, tournament history, statistics, schedules, and
data-driven predictions.

Built with **LangGraph** (multi-agent orchestration), **MCP** (Model Context
Protocol) tools, **Groq** LLM, **RAG** with **ChromaDB**, and a **Streamlit**
frontend deployable to **Streamlit Cloud**.

---

## ✨ Features

- **Multi-agent architecture** — a Supervisor agent classifies intent and routes to
  specialist agents (Team, History, Prediction, Schedule, General).
- **MCP tool layer** — all data access flows through well-defined MCP tools
  (`get_team_info`, `get_worldcup_history`, `get_team_ranking`, `get_matches`,
  `get_schedule`, …) exposed via a FastMCP server *and* used in-process by the agents.
- **RAG over a team knowledge base** — ChromaDB with bundled ONNX MiniLM embeddings
  (no PyTorch required → fast, light Streamlit Cloud deploys).
- **Transparent predictions** — a simple, explainable scoring algorithm using FIFA
  ranking points + historical pedigree. **No gambling / betting functionality.**
- **Modern Streamlit UI** — chat interface, chat history, team selector, example
  prompts, and a live statistics panel.
- **Production hygiene** — centralized `Config`, structured logging, graceful error
  handling, type hints, and a modular package layout.

---

## 🧱 Architecture

```
User
  │
Supervisor Agent  ──►  (intent analysis + routing)
  │
  ├── Team Agent        → RAG (ChromaDB) + get_team_info
  ├── History Agent     → CSV lookup via get_worldcup_history / get_titles_table
  ├── Prediction Agent  → scoring on get_team_ranking + historical performance
  ├── Schedule Agent    → get_schedule (team / group)
  └── General Agent     → grounded free-form answers
  │
 END  (response returned to the user)
```

All specialists call the shared **MCP tools** (`src/mcp/tools.py`). The same tools are
registered on a standalone **MCP server** (`src/mcp/server.py`) for external MCP clients.

### Project layout

```
.
├── app.py                     # Streamlit frontend
├── groq_chatbot.py            # Single LLM interface (Groq) — used everywhere
├── config.py                  # Centralized Config class
├── requirements.txt
├── data/                      # Sample datasets
│   ├── team_knowledge.json    # RAG knowledge base
│   ├── world_cup_history.csv
│   ├── world_cup_matches.csv
│   ├── fifa_rankings.csv
│   └── world_cup_schedule.csv
├── src/
│   ├── datastore.py           # Cached CSV/JSON data-access layer
│   ├── rag/
│   │   ├── vector_store.py     # ChromaDB wrapper
│   │   └── ingest.py           # Ingestion script
│   ├── mcp/
│   │   ├── tools.py            # Tool implementations (single source of truth)
│   │   └── server.py           # FastMCP server registering the tools
│   ├── agents/                 # Supervisor + specialist agents
│   ├── graph/workflow.py       # LangGraph StateGraph wiring
│   └── utils/logging_config.py
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## 🚀 Quick start (local)

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt

# Provide your Groq API key (any one of these):
#   1) environment variable
export GROQ_API_KEY="your-key"          # Windows PowerShell: $env:GROQ_API_KEY="your-key"
#   2) or copy .streamlit/secrets.toml.example -> .streamlit/secrets.toml and fill it in

# (Optional) pre-build the vector store; the app also does this automatically on first run
python -m src.rag.ingest

streamlit run app.py
```

Open http://localhost:8501.

### Run the MCP server standalone (optional)

```bash
python -m src.mcp.server      # stdio transport — connect any MCP client
```

### Try the raw LLM interface

```bash
python groq_chatbot.py        # tiny REPL against the configured Groq model
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo/branch, set **Main file path** to `app.py`.
4. Under **Advanced settings**, choose **Python 3.11** (or newer).
5. Open **Secrets** and paste:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   # optional: GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

6. Click **Deploy**. On first load the app ingests the knowledge base into ChromaDB
   automatically — no Docker, no extra build steps.

> **Note:** The ChromaDB store is rebuilt on each cold start (the `chroma_db/` folder
> is git-ignored and ephemeral on Streamlit Cloud). Ingestion is fast for the bundled
> dataset.

---

## ⚙️ Configuration

All settings live in `config.py` and can be overridden by environment variables /
Streamlit secrets:

| Variable            | Default                     | Description                       |
| ------------------- | --------------------------- | --------------------------------- |
| `GROQ_API_KEY`      | *(fallback in code)*        | Groq API key                      |
| `GROQ_MODEL`        | `llama-3.3-70b-versatile`   | Groq model name                   |
| `CHROMA_DIR`        | `chroma_db`                 | ChromaDB persistence directory    |
| `CHROMA_COLLECTION` | `wc2026_teams`              | Vector collection name            |
| `RAG_TOP_K`         | `4`                         | Retrieved chunks per query        |
| `LOG_LEVEL`         | `INFO`                      | Logging level                     |

---

## 🔌 Extending the system

- **Add data:** drop new rows into the CSVs or teams into `data/team_knowledge.json`,
  then re-run `python -m src.rag.ingest`.
- **Add a tool:** implement it in `src/mcp/tools.py`, add it to the `TOOLS` registry,
  and register it in `src/mcp/server.py`.
- **Add an agent:** create a node in `src/agents/`, add a route keyword/branch in
  `src/agents/supervisor.py`, and wire it into `src/graph/workflow.py`.

---

## 🔐 Security note

`groq_chatbot.py` ships with a hardcoded fallback API key purely so the project runs
out of the box. **Rotate that key and set `GROQ_API_KEY` via secrets** before any real
deployment.

---

## 📜 Disclaimer

Datasets are illustrative samples for demonstration. Predictions are analytical
estimates based on rankings and history — **not betting advice**.

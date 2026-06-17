"""MCP server exposing the World Cup tools over the Model Context Protocol.

Run as a standalone MCP server (stdio transport)::

    python -m src.mcp.server

This registers every tool from :mod:`src.mcp.tools` with a FastMCP server so any
MCP-compatible client (Claude Desktop, IDEs, other agents) can call them. The
Streamlit app itself uses the tools in-process for performance and to stay within
a single Streamlit Cloud process, but the same logic is served here for
spec-compliant external integration.
"""

from __future__ import annotations

from typing import Any, Dict

from src.mcp import tools as wc_tools
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_server() -> Any:
    """Construct and return a FastMCP server with all tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it with: pip install mcp"
        ) from exc

    server = FastMCP("fifa-worldcup-2026")

    @server.tool()
    def get_team_info(team_name: str) -> Dict[str, Any]:
        """Get information about a World Cup team (coach, captain, players, titles)."""
        return wc_tools.get_team_info(team_name)

    @server.tool()
    def get_worldcup_history(year: int) -> Dict[str, Any]:
        """Get the result of a specific World Cup edition by year."""
        return wc_tools.get_worldcup_history(year)

    @server.tool()
    def get_team_ranking(team_name: str) -> Dict[str, Any]:
        """Get the current FIFA ranking for a team."""
        return wc_tools.get_team_ranking(team_name)

    @server.tool()
    def get_matches(team_name: str) -> Dict[str, Any]:
        """Get historical World Cup matches involving a team."""
        return wc_tools.get_matches(team_name)

    @server.tool()
    def get_schedule(team_name: str = "", group: str = "") -> Dict[str, Any]:
        """Get 2026 World Cup fixtures for a team or a group."""
        return wc_tools.get_schedule(team_name=team_name, group=group)

    @server.tool()
    def get_titles_table() -> Dict[str, Any]:
        """Get the table of World Cup titles won per country."""
        return wc_tools.get_titles_table()

    @server.tool()
    def get_top_rankings(n: int = 10) -> Dict[str, Any]:
        """Get the top-n FIFA-ranked teams."""
        return wc_tools.get_top_rankings(n)

    logger.info("MCP server built with tools: %s", wc_tools.list_tools())
    return server


def main() -> None:
    server = build_server()
    server.run()  # stdio transport by default


if __name__ == "__main__":
    main()

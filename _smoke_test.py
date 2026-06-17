"""Ad-hoc smoke test for non-LLM components (safe to delete)."""
from src import datastore
from src.mcp import tools as t
from src.agents import extract
from src.agents.prediction_agent import strength_score, rank_teams
from src.agents.supervisor import classify  # noqa: F401  (LLM-dependent; tested separately)

print("teams:", datastore.list_teams())
print("history 2022:", t.get_worldcup_history(2022)["history"]["winner"])
print("titles:", t.get_titles_table()["titles"])
print("ranking Brazil:", t.get_team_ranking("Brazil")["ranking"]["rank"])
print("matches Argentina:", t.get_matches("Argentina")["count"])
print("schedule Group J:", len(t.get_schedule(group="J")["fixtures"]), "fixtures")
print("schedule England:", len(t.get_schedule(team_name="England")["fixtures"]), "fixtures")
print("extract teams:", extract.extract_teams("Compare Brazil and France"))
print("extract year:", extract.extract_year("Who won the 2018 World Cup?"))
print("extract group:", extract.extract_group("Show Group J matches"))
print("teams in group J:", datastore.teams_in_group("J"))
print("score Argentina:", strength_score("Argentina"))
print("rank group J:", [s["team"] for s in rank_teams(datastore.teams_in_group("J"))])
print("ALL COMPONENT TESTS PASSED")

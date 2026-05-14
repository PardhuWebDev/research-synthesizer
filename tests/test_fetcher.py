import sys
sys.path.append(".")

from graph.state import ResearchState
from agents.fetcher import fetcher_agent

initial_state: ResearchState = {
    "query": "transformer attention mechanisms",
    "papers": [],
    "chunks": [],
    "retrieved": [],
    "claims": {},
    "contradictions": [],
    "report": "",
    "error": None
}

result = fetcher_agent(initial_state)

print(f"\n Total papers: {len(result['papers'])}")
for p in result["papers"]:
    print(f"  [{p['published']}] {p['title'][:70]}")
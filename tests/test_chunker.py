import sys
sys.path.append(".")

from graph.state import ResearchState
from agents.fetcher import fetcher_agent
from agents.chunker import chunker_agent

state: ResearchState = {
    "query": "transformer attention mechanisms",
    "papers": [],
    "chunks": [],
    "retrieved": [],
    "claims": {},
    "contradictions": [],
    "report": "",
    "error": None
}

state = fetcher_agent(state)
state = chunker_agent(state)

print(f"\nTotal chunks: {len(state['chunks'])}")
print(f"Sample chunk:\n{state['chunks'][0][:200]}")
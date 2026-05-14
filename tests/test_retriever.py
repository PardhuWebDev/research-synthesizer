import sys
sys.path.append(".")

from graph.state import ResearchState
from agents.fetcher import fetcher_agent
from agents.chunker import chunker_agent
from agents.retriever import retriever_agent

state: ResearchState = {
    "query": "how does attention mechanism work in transformers",
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
state = retriever_agent(state)

print(f"\nTop retrieved chunk:\n{state['retrieved'][0][:300]}")
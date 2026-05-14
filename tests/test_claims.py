import sys
sys.path.append(".")

from graph.state import ResearchState
from agents.fetcher import fetcher_agent
from agents.chunker import chunker_agent
from agents.retriever import retriever_agent
from agents.claim_extractor import claim_extractor_agent

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
state = retriever_agent(state)
state = claim_extractor_agent(state)

print(f"\nTotal papers with claims: {len(state['claims'])}")
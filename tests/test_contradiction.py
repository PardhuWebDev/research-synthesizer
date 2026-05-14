import sys
sys.path.append(".")

from graph.state import ResearchState
from agents.contradiction import contradiction_agent

state: ResearchState = {
    "query": "transformer attention",
    "papers": [
        {"id": "paper_001", "title": "Paper A", "authors": [], "abstract": "", "published": "", "url": ""},
        {"id": "paper_002", "title": "Paper B", "authors": [], "abstract": "", "published": "", "url": ""},
    ],
    "chunks": [],
    "retrieved": [],
    "claims": {
        "paper_001": [
            "Self-attention scales quadratically with sequence length.",
            "Transformer models require large amounts of training data.",
        ],
        "paper_002": [
            "Linear attention achieves O(n) complexity with no performance loss.",
            "Transformer models can be trained effectively on small datasets.",
        ]
    },
    "contradictions": [],
    "report": "",
    "error": None
}

result = contradiction_agent(state)

print(f"\nTotal contradictions found: {len(result['contradictions'])}")
for c in result["contradictions"]:
    print(f"\n[{c['confidence']}] score={c['score']}")
    print(f"  A: {c['claim_a']}")
    print(f"  B: {c['claim_b']}")
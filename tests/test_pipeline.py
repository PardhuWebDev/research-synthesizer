import sys
sys.path.append(".")

from graph.graph import pipeline
from graph.state import ResearchState

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

print("=" * 60)
print("RESEARCH SYNTHESIZER — FULL PIPELINE RUN")
print("=" * 60)

result = pipeline.invoke(initial_state)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result["report"])

print("\n" + "=" * 60)
print(f"CONTRADICTIONS FOUND: {len(result['contradictions'])}")
print("=" * 60)
for c in result["contradictions"]:
    print(f"\n[{c['confidence']}] score={c['score']}")
    print(f"  A ({c['paper_a_title'][:40]}): {c['claim_a'][:70]}")
    print(f"  B ({c['paper_b_title'][:40]}): {c['claim_b'][:70]}")
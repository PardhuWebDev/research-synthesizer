import os
from dotenv import load_dotenv
from groq import Groq
from graph.state import ResearchState

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def synthesizer_agent(state: ResearchState) -> ResearchState:
    print("[Synthesizer] Generating research report...")

    context = "\n\n".join(state["retrieved"][:6])

    contradictions_text = ""
    if state["contradictions"]:
        contradictions_text = "\n\nDetected Contradictions:\n"
        for c in state["contradictions"]:
            contradictions_text += (
                f"\n[{c['confidence']} confidence | score={c['score']}]\n"
                f"  Paper A: {c['paper_a_title']}\n"
                f"  Claim:   {c['claim_a']}\n"
                f"  Paper B: {c['paper_b_title']}\n"
                f"  Claim:   {c['claim_b']}\n"
            )
    else:
        contradictions_text = "\n\nNo contradictions detected."

    prompt = f"""You are a research synthesis expert.

Based on the retrieved context and detected contradictions below, write a structured research report.

Query: {state['query']}

Retrieved Context:
{context}
{contradictions_text}

Write the report in this exact format:

## Overview
2-3 sentences summarizing the research area.

## Key Findings
3-5 bullet points of the most important findings across the papers.

## Contradictions & Conflicts
Explain any contradictions found. If none, state that the literature is consistent.

## Open Questions
2-3 unresolved questions or gaps in the literature.

## Conclusion
1-2 sentences wrapping up.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    report = response.choices[0].message.content.strip()
    print("[Synthesizer] Report generated.")
    return {**state, "report": report}
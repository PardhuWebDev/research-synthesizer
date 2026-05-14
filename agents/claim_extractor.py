import os
import time
from dotenv import load_dotenv
from groq import Groq
from graph.state import ResearchState

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_claims(title: str, abstract: str) -> list[str]:
    prompt = f"""You are a scientific claim extractor.

Given the title and abstract of a research paper, extract exactly 3 to 5 key falsifiable claims.

Rules:
- Each claim must be a single, atomic, factual assertion
- Claims must be specific and verifiable, not vague summaries
- Do NOT include opinions, contributions, or future work
- Return ONLY a numbered list, nothing else

Title: {title}
Abstract: {abstract}

Claims:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    lines = response.choices[0].message.content.strip().split("\n")
    claims = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            claim = line.split(".", 1)[-1].strip()
            if claim:
                claims.append(claim)
    return claims[:5]


def claim_extractor_agent(state: ResearchState) -> ResearchState:
    print(f"[ClaimExtractor] Extracting claims from {len(state['papers'])} papers...")

    all_claims: dict[str, list[str]] = {}

    for i, paper in enumerate(state["papers"]):
        try:
            print(f"  [{i+1}/{len(state['papers'])}] {paper['title'][:55]}...")
            claims = extract_claims(paper["title"], paper["abstract"])
            all_claims[paper["id"]] = claims
            print(f"     -> {len(claims)} claims extracted")
            for j, c in enumerate(claims, 1):
                print(f"        {j}. {c[:80]}")
            time.sleep(1)
        except Exception as e:
            print(f"     -> ERROR: {e}")
            all_claims[paper["id"]] = []

    print(f"[ClaimExtractor] Done.")
    return {**state, "claims": all_claims}
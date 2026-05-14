from sentence_transformers import SentenceTransformer, CrossEncoder
from itertools import combinations
from graph.state import ResearchState, ConflictRecord
import uuid

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

SIMILARITY_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.75


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x**2 for x in a) ** 0.5
    mag_b = sum(x**2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_confidence(score: float) -> str:
    if score >= 0.90:
        return "HIGH"
    elif score >= 0.75:
        return "MEDIUM"
    return "LOW"


def contradiction_agent(state: ResearchState) -> ResearchState:
    print("[ContradictionDetector] Starting detection...")

    claims = state["claims"]
    papers = {p["id"]: p for p in state["papers"]}

    all_claims = []
    for paper_id, claim_list in claims.items():
        for claim in claim_list:
            all_claims.append((paper_id, claim))

    print(f"  Total claims: {len(all_claims)}")

    texts = [c[1] for c in all_claims]
    if len(texts) < 2:
        print("  Not enough claims for comparison.")
        return {**state, "contradictions": []}

    embeddings = embedder.encode(texts)

    candidates = []
    for (i, (pid_a, claim_a)), (j, (pid_b, claim_b)) in combinations(enumerate(all_claims), 2):
        if pid_a == pid_b:
            continue
        sim = cosine_similarity(embeddings[i].tolist(), embeddings[j].tolist())
        if sim >= SIMILARITY_THRESHOLD:
            candidates.append((pid_a, claim_a, pid_b, claim_b, sim))

    print(f"  Candidate pairs after similarity filter: {len(candidates)}")

    contradictions: list[ConflictRecord] = []

    for pid_a, claim_a, pid_b, claim_b, sim in candidates:
        scores = nli_model.predict([(claim_a, claim_b)])
        import math
        logits = scores[0]
        exp_scores = [math.exp(x) for x in logits]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        contradiction_score = float(probs[0])
        if contradiction_score >= CONTRADICTION_THRESHOLD:
            confidence = get_confidence(contradiction_score)
            paper_a = papers.get(pid_a, {})
            paper_b = papers.get(pid_b, {})

            record = ConflictRecord(
                id=str(uuid.uuid4())[:8],
                confidence=confidence,
                score=round(contradiction_score, 3),
                paper_a_id=pid_a,
                paper_a_title=paper_a.get("title", ""),
                claim_a=claim_a,
                paper_b_id=pid_b,
                paper_b_title=paper_b.get("title", ""),
                claim_b=claim_b,
                context=f"Semantic similarity: {sim:.3f}"
            )
            contradictions.append(record)
            print(f"  CONTRADICTION [{confidence}] score={contradiction_score:.3f}")
            print(f"    A: {claim_a[:70]}")
            print(f"    B: {claim_b[:70]}")

    print(f"[ContradictionDetector] Found {len(contradictions)} contradictions.")
    return {**state, "contradictions": contradictions}
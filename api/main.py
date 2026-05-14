import sys
sys.path.append(".")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.graph import pipeline
from graph.state import ResearchState

app = FastAPI(
    title="Research Synthesizer API",
    description="Multi-agent system for scientific literature analysis with contradiction detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str
    num_papers: int = 10


class ContradictionResponse(BaseModel):
    id: str
    confidence: str
    score: float
    paper_a_title: str
    claim_a: str
    paper_b_title: str
    claim_b: str


class PaperResponse(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    url: str


class ResearchResponse(BaseModel):
    query: str
    papers_analyzed: int
    claims_extracted: int
    contradictions_found: int
    contradictions: list[ContradictionResponse]
    papers: list[PaperResponse]
    report: str


@app.get("/")
def root():
    return {"status": "running", "project": "Autonomous Research Synthesizer"}


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    try:
        initial_state: ResearchState = {
            "query": request.query,
            "papers": [],
            "chunks": [],
            "retrieved": [],
            "claims": {},
            "contradictions": [],
            "report": "",
            "error": None
        }

        result = pipeline.invoke(initial_state)
        total_claims = sum(len(c) for c in result["claims"].values())

        contradictions = [
            ContradictionResponse(
                id=c["id"],
                confidence=c["confidence"],
                score=c["score"],
                paper_a_title=c["paper_a_title"],
                claim_a=c["claim_a"],
                paper_b_title=c["paper_b_title"],
                claim_b=c["claim_b"]
            )
            for c in result["contradictions"]
        ]

        papers = [
            PaperResponse(
                id=str(p.get("id", "")),
                title=str(p.get("title", "")),
                authors=list(p.get("authors", [])),
                abstract=str(p.get("abstract", "")),
                published=str(p.get("published", "")),
                url=str(p.get("url", ""))
            )
            for p in result["papers"]
        ]
        return ResearchResponse(
            query=result["query"],
            papers_analyzed=len(result["papers"]),
            claims_extracted=total_claims,
            contradictions_found=len(contradictions),
            contradictions=contradictions,
            papers=papers,
            report=result["report"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}
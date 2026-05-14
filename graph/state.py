from typing import TypedDict, List, Dict, Optional


class PaperMeta(TypedDict):
    id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    url: str


class ConflictRecord(TypedDict):
    id: str
    confidence: str
    score: float
    paper_a_id: str
    paper_a_title: str
    claim_a: str
    paper_b_id: str
    paper_b_title: str
    claim_b: str
    context: str


class ResearchState(TypedDict):
    query: str
    papers: List[PaperMeta]
    chunks: List[str]
    retrieved: List[str]
    claims: Dict[str, List[str]]
    contradictions: List[ConflictRecord]
    report: str
    error: Optional[str]
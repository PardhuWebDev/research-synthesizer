import arxiv
from graph.state import ResearchState, PaperMeta


def fetcher_agent(state: ResearchState) -> ResearchState:
    print(f"[Fetcher] Searching Arxiv for: {state['query']}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=state["query"],
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers: list[PaperMeta] = []

    for result in client.results(search):
        papers.append(PaperMeta(
            id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary,
            published=str(result.published.date()),
            url=result.entry_id
        ))
        print(f"  -> Fetched: {result.title[:60]}...")

    print(f"[Fetcher] Total papers fetched: {len(papers)}")

    return {**state, "papers": papers}
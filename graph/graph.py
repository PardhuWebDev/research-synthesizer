from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from agents.fetcher import fetcher_agent
from agents.chunker import chunker_agent
from agents.retriever import retriever_agent
from agents.claim_extractor import claim_extractor_agent
from agents.contradiction import contradiction_agent
from agents.synthesizer import synthesizer_agent


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("fetcher",     fetcher_agent)
    graph.add_node("chunker",     chunker_agent)
    graph.add_node("retriever",   retriever_agent)
    graph.add_node("extractor",   claim_extractor_agent)
    graph.add_node("detector",    contradiction_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.set_entry_point("fetcher")
    graph.add_edge("fetcher",     "chunker")
    graph.add_edge("chunker",     "retriever")
    graph.add_edge("retriever",   "extractor")
    graph.add_edge("extractor",   "detector")
    graph.add_edge("detector",    "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


pipeline = build_graph()
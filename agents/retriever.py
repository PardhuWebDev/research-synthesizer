from sentence_transformers import SentenceTransformer
import chromadb
from graph.state import ResearchState

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./store/chroma_db")


def retriever_agent(state: ResearchState) -> ResearchState:
    print(f"[Retriever] Retrieving relevant chunks for: {state['query']}")

    collection = chroma_client.get_or_create_collection(
        name="research_papers",
        metadata={"hnsw:space": "cosine"}
    )

    query_embedding = model.encode(state["query"]).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(10, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append(f"[{meta['title']}]\n{doc}")
        print(f"  -> Retrieved (score {1-dist:.3f}): {meta['title'][:55]}...")

    print(f"[Retriever] Total chunks retrieved: {len(retrieved)}")
    return {**state, "retrieved": retrieved}
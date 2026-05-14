from sentence_transformers import SentenceTransformer
import chromadb
from graph.state import ResearchState

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./store/chroma_db")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def chunker_agent(state: ResearchState) -> ResearchState:
    print(f"[Chunker] Processing {len(state['papers'])} papers...")

    collection = chroma_client.get_or_create_collection(
        name="research_papers",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []

    for paper in state["papers"]:
        text = f"{paper['title']}\n\n{paper['abstract']}"
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{paper['id']}_chunk_{i}"
            embedding = model.encode(chunk).tolist()

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "chunk_index": i
                }]
            )
            all_chunks.append(chunk)

        print(f"  -> Chunked & embedded: {paper['title'][:55]}... ({len(chunks)} chunks)")

    print(f"[Chunker] Total chunks stored: {len(all_chunks)}")
    return {**state, "chunks": all_chunks}
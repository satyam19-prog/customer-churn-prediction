import os
from langchain_community.vectorstores import Chroma
from backend.rag.embeddings import get_embedding_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_STORE_DIR = os.path.join(BASE_DIR, 'backend', 'rag', 'chroma_store')

def get_retention_strategies(query: str, risk_level: str, k: int = 6) -> list[str]:
    """
    Args:
        query: constructed retrieval query string
        risk_level: one of "Low", "Medium", "High", "Critical"
        k: number of chunks to return (default 6)
    
    Returns:
        List of strings, each being a relevant text chunk
    """
    if not os.path.exists(CHROMA_STORE_DIR):
        raise FileNotFoundError("ChromaDB not found. Run: python backend/rag/build_vectorstore.py")

    vectorstore = Chroma(
        persist_directory=CHROMA_STORE_DIR,
        embedding_function=get_embedding_model(),
        collection_name="retention_strategies"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k * 2,
            "fetch_k": 20,
            "lambda_mult": 0.6,
        }
    )

    retrieved_docs = retriever.invoke(query)
    
    filtered_chunks = []
    for doc in retrieved_docs:
        risk_levels = doc.metadata.get("risk_levels", [])
        if risk_level in risk_levels:
            filtered_chunks.append(doc.page_content)
            if len(filtered_chunks) == k:
                break
                
    return filtered_chunks

if __name__ == "__main__":
    results = get_retention_strategies(
        query="month-to-month contract high charges electronic check payment",
        risk_level="High",
        k=4
    )
    for i, r in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(r[:200])
        print()

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns HuggingFace all-MiniLM-L6-v2 embedding model.
    384 dimensions, free, no API key required.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

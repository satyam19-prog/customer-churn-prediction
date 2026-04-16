import streamlit as st

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def get_embedding_model():
    """
    Returns HuggingFace all-MiniLM-L6-v2 embedding model.
    Cached to prevent reloading the model on every inference.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

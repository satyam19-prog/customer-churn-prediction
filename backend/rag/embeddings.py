import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

@st.cache_resource
def get_embedding_model():
    """
    Returns FastEmbed all-MiniLM-L6-v2 embedding model.
    Cached to prevent reloading the ONNX model on every inference.
    Massively reduces RAM requirements by removing PyTorch.
    """
    return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

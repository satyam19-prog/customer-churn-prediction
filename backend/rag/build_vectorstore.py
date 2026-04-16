import os
from collections import Counter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from backend.rag.embeddings import get_embedding_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, 'backend', 'rag', 'knowledge_base')
CHROMA_STORE_DIR = os.path.join(BASE_DIR, 'backend', 'rag', 'chroma_store')

def build_index():
    print(f"Loading documents from {KNOWLEDGE_BASE_DIR}...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    
    print(f"Loaded {len(docs)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n- ", "\n", " "]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    risk_mapping = {
        "telecom_churn_strategies.md": ["Low", "Medium", "High", "Critical"],
        "contract_retention_playbook.md": ["Medium", "High", "Critical"],
        "payment_risk_signals.md": ["High", "Critical"],
        "service_upsell_matrix.md": ["Low", "Medium", "High"],
        "ethical_ai_retention.md": ["Low", "Medium", "High", "Critical"],
        "industry_benchmarks.md": ["Low", "Medium", "High", "Critical"]
    }
    
    source_counts = Counter()
    
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        filename = os.path.basename(source_path)
        
        # update metadata
        chunk.metadata["source"] = filename
        
        # Assign risk levels based on filename
        if filename in risk_mapping:
            chunk.metadata["risk_levels"] = risk_mapping[filename]
        else:
            chunk.metadata["risk_levels"] = ["Low", "Medium", "High", "Critical"]
            
        source_counts[filename] += 1

    print(f"\nTotal chunks to index: {len(chunks)}")
    print("Chunks per source file:")
    for filename, count in source_counts.items():
        print(f"  - {filename}: {count} chunks")
        
    print("\nBuilding ChromaDB index...")
    
    os.makedirs(CHROMA_STORE_DIR, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=CHROMA_STORE_DIR,
        collection_name="retention_strategies"
    )
    
    print("Index built successfully and persisted to:", CHROMA_STORE_DIR)

if __name__ == "__main__":
    build_index()

# RetainAI — Agentic Customer Churn Intelligence

## Overview
RetainAI is a production-grade, end-to-end customer churn prediction system that fuses traditional machine learning with agentic AI reasoning. By orchestrating a LangGraph multi-agent workflow over a retrieval-augmented generation (RAG) architecture, the platform moves beyond simple churn probability scores to actively generate, validate, and suggest highly personalized, contextual retention strategies.

The system is built with a sleek Streamlit frontend, a robust Scikit-learn ML core with SHAP explainers, and a ChromaDB-powered local vector database. Utilizing state-of-the-art LLMs, RetainAI effectively formulates communication playbooks rooted in telecom benchmarks, ethical AI practices, and past interaction feedback loops.

## Architecture
```text
[UI Layer: Streamlit Premium Interface]
        ↓
[ML Layer: LR/DT + SHAP Explainer]
        ↓
[LangGraph Agent: 5-Node Workflow]
    ├── risk_analyzer
    ├── context_retriever ←→ [ChromaDB RAG: 6 Knowledge Docs]
    ├── strategy_reasoner ←→ [LLM: Groq llama-3.1-8b / Gemini Flash]
    ├── plan_generator
    └── validator (retry loop)
        ↓
[Structured JSON Retention Plan]
```

## Quickstart
### 1. Clone & install
```bash
git clone https://github.com/satyam19-prog/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Set up API keys
Copy `.env.example` to `.env` and fill `GROQ_API_KEY`:
```bash
cp .env.example .env
```

### 3. Build vector store
```bash
python backend/rag/build_vectorstore.py
```

### 4. Run app
```bash
streamlit run frontend/app.py
```

## Project Structure
- `frontend/app.py`: Premium Streamlit dashboard for Single, Batch, and Scenario simulator analyses.
- `backend/feedback.py`: Core logic for recording ratings and routing negative behaviors back into the MMR vector queries.
- `backend/models/predict.py`: Scikit-learn inference pipeline with integrated SHAP explainability.
- `backend/workflows/retention_graph.py`: LangGraph StateGraph connecting all agent nodes into an executable pipeline.
- `backend/agents/state.py`: TypedDict defining the complex LangGraph execution state parameters.
- `backend/agents/risk_analyzer.py`: Initial agent node mapping raw churn probabilities into concrete risk tiers.
- `backend/agents/context_retriever.py`: Integrates dynamic search queries with the RAG ChromaDB system.
- `backend/agents/strategy_reasoner.py`: Interacts with LLM models to generate comprehensive reasoning traces.
- `backend/agents/plan_generator.py`: Parses reasoning output chunks into strict JSON implementations.
- `backend/agents/validator.py`: Asserts correct JSON structures via looping retries against the generator.
- `backend/rag/embeddings.py`: Configures HuggingFace `all-MiniLM-L6-v2` embedding engine interface.
- `backend/rag/build_vectorstore.py`: Directory parser storing textual retention knowledge baselines into ChromaDB.
- `backend/rag/retriever.py`: Exposes MMR diversity search algorithms across the embedded vector knowledge base.
- `backend/rag/knowledge_base/`: Directory containing targeted markdown metrics supporting the Retrieval-Augmented Generation context.

## Agent Workflow
- **Risk Analyzer**: Categorizes churn probabilities into critical risk tiers while analyzing overarching variable structures.
- **Context Retriever**: Fetches contextually diverse chunks of telecom retention data based on the extracted risk profiles and explicit user constraints.
- **Strategy Reasoner**: Produces holistic cognitive traces connecting SHAP driver outputs linearly with contextual playbook facts.
- **Plan Generator**: Distills the large reasoning narratives directly into predictable, rigid JSON action arrays.
- **Validator**: Guarantees JSON schema fidelity universally, gating broken formats mapping backward loops within the graph network.

## RAG Pipeline
Specialized markdown directories containing specific telecommunications benchmarks, ethical guidelines, and playbook semantics construct our foundational context mechanism. The textual pipeline is parsed systematically into uniformly sized embeddings configured via localized HuggingFace CPU transformers. Querying the constructed ChromaDB retrieval mechanisms guarantees variance via MMR algorithms natively factoring in our negative reinforcement feedback loop vectors natively gathered through UI iterations.

## Live Demo
[STREAMLIT_DEPLOY_URL] | [HF_DEPLOY_URL]

## Team
[Your team names]

## Ethical AI Notice
In absolute alignment with the internal `ethical_ai_retention.md` directives, RetainAI strictly advocates toward unbiased customer engagements void of predatory structures. Outputs automatically indicate AI generation and explicitly highlight that final strategies remain subject entirely to human authorization.

from backend.agents.state import AgentState
from backend.rag.retriever import get_retention_strategies

def context_retriever_node(state: AgentState) -> AgentState:
    query = state.get("retrieval_query", "")
    risk_level = state.get("risk_level", "Medium")
    
    retrieved = get_retention_strategies(query, risk_level, k=6)
    state["retrieved_docs"] = retrieved
    
    return state

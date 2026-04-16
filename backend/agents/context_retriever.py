from backend.agents.state import AgentState
from backend.rag.retriever import get_retention_strategies
from backend.feedback import get_low_rated_strategies

def context_retriever_node(state: AgentState) -> AgentState:
    query = state.get("retrieval_query", "")
    risk_level = state.get("risk_level", "Medium")
    
    low_rated = get_low_rated_strategies(risk_level)
    if low_rated:
        query = f"AVOID strategies similar to: {', '.join(low_rated)}. INSTEAD FOCUS ON: " + query
        state["retrieval_query"] = query
        
    retrieved = get_retention_strategies(query, risk_level, k=6)
    state["retrieved_docs"] = retrieved
    
    return state

import os
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.risk_analyzer import risk_analyzer_node
from backend.agents.context_retriever import context_retriever_node
from backend.agents.strategy_reasoner import strategy_reasoner_node
from backend.agents.plan_generator import plan_generator_node
from backend.agents.validator import validator_node

# 1. Create StateGraph
graph = StateGraph(AgentState)

# 2. Add nodes
graph.add_node("risk_analyzer", risk_analyzer_node)
graph.add_node("context_retriever", context_retriever_node)
graph.add_node("strategy_reasoner", strategy_reasoner_node)
graph.add_node("plan_generator", plan_generator_node)
graph.add_node("validator", validator_node)

# 3. Set entry
graph.set_entry_point("risk_analyzer")

# 4. Add linear edges
graph.add_edge("risk_analyzer", "context_retriever")
graph.add_edge("context_retriever", "strategy_reasoner")
graph.add_edge("strategy_reasoner", "plan_generator")
graph.add_edge("plan_generator", "validator")

# 5. Add conditional edge on validator
def validator_router(state: AgentState) -> str:
    if state.get("validation_passed", False) or state.get("iteration_count", 0) >= 2:
        return END
    else:
        return "plan_generator"

graph.add_conditional_edges(
    "validator",
    validator_router,
    {
        END: END,
        "plan_generator": "plan_generator"
    }
)

# 6. Compile
retention_agent = graph.compile()

# 7. Create a wrapper function
def run_retention_agent(customer_data: dict, churn_result: dict) -> dict:
    """
    Args:
        customer_data: raw customer feature dict
        churn_result: output from backend.models.predict.predict_churn()
    Returns:
        retention_plan dict, or error dict on failure
    """
    initial_state = AgentState(
        customer_data=customer_data,
        churn_probability=churn_result.get("churn_probability", 0.0),
        churn_prediction=churn_result.get("churn_prediction", 0),
        shap_values=churn_result.get("shap_values", {}),
        risk_level="",
        risk_summary="",
        top_churn_drivers=[],
        retrieval_query="",
        retrieved_docs=[],
        reasoning_trace="",
        retention_plan={},
        validation_passed=False,
        iteration_count=0,
        error_message=None
    )
    try:
        final_state = retention_agent.invoke(initial_state)
        return final_state["retention_plan"]
    except Exception as e:
        return {"error": str(e), "priority_level": "Unknown",
                "recommended_actions": [], "risk_analysis": "Agent error"}

# 8. Add end-to-end test at bottom
if __name__ == "__main__":
    from backend.models.predict import predict_churn
    test_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5, "TotalCharges": 179.0
    }
    churn_result = predict_churn(test_customer)
    print("ML Result:", churn_result["churn_probability"], churn_result["risk_level"] if "risk_level" in churn_result else "")
    plan = run_retention_agent(test_customer, churn_result)
    import json
    print(json.dumps(plan, indent=2))

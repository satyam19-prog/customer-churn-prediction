from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    customer_data: Dict[str, Any]
    churn_probability: float
    churn_prediction: int
    shap_values: Dict[str, float]
    risk_level: str           # "Low" | "Medium" | "High" | "Critical"
    risk_summary: str
    top_churn_drivers: List[str]  # top 5 feature names by abs SHAP
    retrieval_query: str
    retrieved_docs: List[str]
    reasoning_trace: str
    retention_plan: Dict[str, Any]
    validation_passed: bool
    iteration_count: int
    error_message: Optional[str]

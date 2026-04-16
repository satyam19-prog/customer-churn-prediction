from backend.agents.state import AgentState

def risk_analyzer_node(state: AgentState) -> AgentState:
    prob = state.get("churn_probability", 0.0)
    
    if prob < 0.35:
        risk_level = "Low"
    elif prob <= 0.60:
        risk_level = "Medium"
    elif prob <= 0.80:
        risk_level = "High"
    else:
        risk_level = "Critical"
        
    state["risk_level"] = risk_level
    
    shap_vals = state.get("shap_values", {})
    sorted_shap = sorted(shap_vals.keys(), key=lambda k: abs(shap_vals[k]), reverse=True)
    
    top_drivers = []
    for k in sorted_shap[:5]:
        parts = k.split("__")
        clean_name = parts[-1] if len(parts) > 1 else k
        
        if "_" in clean_name:
            clean_name = clean_name.replace("_", ": ", 1)
            
        top_drivers.append(clean_name)
        
    state["top_churn_drivers"] = top_drivers
    
    drivers_str = ", ".join(top_drivers[:3]) if top_drivers else "None"
    summary = f"Customer has {risk_level} churn risk ({prob:.1%} probability). Primary drivers: {drivers_str}."
    state["risk_summary"] = summary
    
    customer_data = state.get("customer_data", {})
    contract = customer_data.get("Contract", "Unknown")
    tenure = customer_data.get("tenure", "Unknown")
    monthly_charges = customer_data.get("MonthlyCharges", "Unknown")
    
    query = f"{risk_level} risk customer. {drivers_str}. Contract: {contract}. Tenure: {tenure} months. Monthly charges: ${monthly_charges}."
    state["retrieval_query"] = query
    
    return state

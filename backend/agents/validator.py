from backend.agents.state import AgentState

def validator_node(state: AgentState) -> AgentState:
    plan = state.get("retention_plan", {})
    failed = []
    
    if not isinstance(plan, dict) or not plan:
        failed.append("retention_plan is empty or not a dict")
    else:
        if not plan.get("recommended_actions") or len(plan["recommended_actions"]) == 0:
            failed.append("recommended_actions key missing or empty")
            
        priority = plan.get("priority_level", "")
        if priority not in ["Immediate", "Within 7 days", "Within 30 days"]:
            failed.append(f"priority_level '{priority}' is invalid")
            
        if "customer_profile" not in plan:
            failed.append("customer_profile key missing")
            
        if "ethical_disclaimer" not in plan:
            failed.append("ethical_disclaimer key missing")
            
    if failed:
        state["validation_passed"] = False
        state["error_message"] = f"Validation failed: {', '.join(failed)}"
    else:
        state["validation_passed"] = True
        state["error_message"] = None
        
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return state

import os
import json
import re
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState

load_dotenv()

def get_llm():
    try:
        from langchain_groq import ChatGroq
        if os.environ.get("GROQ_API_KEY"):
            return ChatGroq(model_name="llama-3.1-8b-instant")
    except ImportError:
        pass
        
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
def run_llm_plan(llm, prompt_text: str):
    messages = [
        SystemMessage(content="You are a retention strategy AI. Output ONLY valid JSON. No markdown. No explanation."),
        HumanMessage(content=prompt_text)
    ]
    response = llm.invoke(messages)
    return response.content

def plan_generator_node(state: AgentState) -> AgentState:
    customer_data = state.get("customer_data", {})
    risk_level = state.get("risk_level", "Unknown")
    try:
        llm = get_llm()
        
        churn_prob = state.get("churn_probability", 0.0)
        reasoning_trace = state.get("reasoning_trace", "")
        
        prompt_text = f"Based on this analysis, generate a structured retention plan.\n\nREASONING:\n{reasoning_trace}\n\nCUSTOMER:\n- Contract: {customer_data.get('Contract', 'Unknown')}, Tenure: {customer_data.get('tenure', 'Unknown')} months\n- Monthly Charges: ${customer_data.get('MonthlyCharges', 'Unknown')}\n- Internet Service: {customer_data.get('InternetService', 'Unknown')}\n- Payment: {customer_data.get('PaymentMethod', 'Unknown')}\n- Risk: {risk_level} ({churn_prob:.1%})\n\nOutput this exact JSON structure:\n{{\n  \"customer_profile\": {{\n    \"tenure_months\": <int>,\n    \"contract_type\": \"<string>\",\n    \"monthly_charges\": <float>,\n    \"risk_tier\": \"<Low|Medium|High|Critical>\"\n  }},\n  \"risk_analysis\": \"<2 sentence summary>\",\n  \"churn_drivers\": [\"<driver1>\", \"<driver2>\", \"<driver3>\"],\n  \"recommended_actions\": [\n    {{\n      \"action\": \"<specific action>\",\n      \"rationale\": \"<why this addresses a churn driver>\",\n      \"channel\": \"<Email|Call|App|SMS>\",\n      \"timeline\": \"<Immediate|Within 7 days|Within 30 days>\"\n    }}\n  ],\n  \"priority_level\": \"<Immediate|Within 7 days|Within 30 days>\",\n  \"expected_impact\": \"<e.g. 35% churn risk reduction>\",\n  \"sources\": [\"<reference from retrieved docs>\"],\n  \"ethical_disclaimer\": \"AI-generated plan. Human review recommended before customer contact.\"\n}}"

        content = run_llm_plan(llm, prompt_text)
        
        clean_json = re.sub(r'```json\s*', '', content)
        clean_json = re.sub(r'```\s*', '', clean_json).strip()
        
        parsed_plan = json.loads(clean_json)
        state["retention_plan"] = parsed_plan
        
    except Exception as e:
        priority = "Immediate" if risk_level in ["Critical", "High"] else "Within 7 days"
        
        # Safely parse numeric types for fallback
        try:
            tenure_val = int(customer_data.get("tenure", 0))
        except:
            tenure_val = 0
            
        try:
            charges_val = float(customer_data.get("MonthlyCharges", 0.0))
        except:
            charges_val = 0.0
            
        state["retention_plan"] = {
            "customer_profile": {
                "tenure_months": tenure_val,
                "contract_type": str(customer_data.get("Contract", "Unknown")),
                "monthly_charges": charges_val,
                "risk_tier": risk_level
            },
            "risk_analysis": "Fallback rule-based assessment. AI analysis failed.",
            "churn_drivers": state.get("top_churn_drivers", [])[:3],
            "recommended_actions": [
                {
                    "action": "Review account manually for retention offer.",
                    "rationale": "High risk detected in fallback rule engine.",
                    "channel": "Call",
                    "timeline": priority
                }
            ],
            "priority_level": priority,
            "expected_impact": "Unknown. Rule-based fallback.",
            "sources": [],
            "ethical_disclaimer": "AI-generated plan. Human review recommended before customer contact."
        }
    
    return state

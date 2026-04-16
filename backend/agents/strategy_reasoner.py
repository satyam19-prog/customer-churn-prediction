import os
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
def run_llm_reasoning(llm, prompt_text: str):
    messages = [
        SystemMessage(content="You are a senior telecom customer retention analyst."),
        HumanMessage(content=prompt_text)
    ]
    response = llm.invoke(messages)
    return response.content

def strategy_reasoner_node(state: AgentState) -> AgentState:
    try:
        llm = get_llm()
        
        churn_prob = state.get("churn_probability", 0.0)
        risk_level = state.get("risk_level", "Unknown")
        top_churn_drivers = state.get("top_churn_drivers", [])
        customer_data = state.get("customer_data", {})
        retrieved_docs = state.get("retrieved_docs", [])
        
        drivers_list = chr(10).join(f"  * {d}" for d in top_churn_drivers)
        docs_list = chr(10).join(f"[Doc {i+1}]: {doc[:300]}" for i, doc in enumerate(retrieved_docs))
        
        prompt_text = f"CUSTOMER RISK PROFILE:\n- Churn Probability: {churn_prob:.1%}\n- Risk Level: {risk_level}\n- Primary Churn Drivers:\n{drivers_list}\n- Contract: {customer_data.get('Contract', 'Unknown')}\n- Tenure: {customer_data.get('tenure', 'Unknown')} months  \n- Monthly Charges: ${customer_data.get('MonthlyCharges', 'Unknown')}\n- Payment Method: {customer_data.get('PaymentMethod', 'Unknown')}\n- Internet Service: {customer_data.get('InternetService', 'Unknown')}\n\nRETRIEVED RETENTION CONTEXT:\n{docs_list}\n\nReason step-by-step:\n1. Which churn drivers are most actionable RIGHT NOW for this specific customer?\n2. Which retrieved strategies best match these drivers? Quote the specific tactic.\n3. What is the likely customer sentiment given their profile?\n\nFormat as:\n<reasoning>\n[your analysis]\n</reasoning>\n\nIMPORTANT: Only reference tactics mentioned in the retrieved context above.\nMaximum 250 words."

        content = run_llm_reasoning(llm, prompt_text)
        
        match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        if match:
            state["reasoning_trace"] = match.group(1).strip()
        else:
            state["reasoning_trace"] = content.strip()
            
    except Exception as e:
        state["reasoning_trace"] = "Reasoning unavailable — using rule-based fallback"
        
    return state

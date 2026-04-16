import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import sys
import os
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from backend.models.predict import predict_churn
from backend.workflows.retention_graph import run_retention_agent

# SECTION 1 — Page config and CSS
st.set_page_config(
    page_title="RetainAI — Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f1117;
    color: #e0e0e0;
}
.metric-card {
    background-color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    margin-bottom: 10px;
}
.stButton > button {
    background-color: #0f1117;
    color: white;
    width: 100%;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    font-size: 16px;
    font-weight: bold; /* Bold active tab usually handled by Streamlit natively but making all bold here for emphasis */
}
.stTabs [aria-selected="true"] {
    font-weight: 800;
}
.badge-critical { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; background-color: #FF4444; color: white; }
.badge-high { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; background-color: #FF8C00; color: white; }
.badge-medium { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; background-color: #FFD700; color: black; }
.badge-low { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; background-color: #00C851; color: white; }
.channel-call { background-color: #007bff; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; }
.channel-email { background-color: #28a745; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; }
.channel-sms { background-color: #fd7e14; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; }
.channel-app { background-color: #6f42c1; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; }
.action-card { background-color: #f8f9fa; border-left: 4px solid #0f1117; padding: 12px; margin-bottom: 10px; border-radius: 4px; }
.impact-box { border: 2px solid #00C851; padding: 10px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

def get_risk_html(risk_level):
    cls = ""
    label = risk_level.upper() + " RISK"
    if risk_level == "Critical": cls = 'badge-critical'
    elif risk_level == "High": cls = 'badge-high'
    elif risk_level == "Medium": cls = 'badge-medium'
    else: cls = 'badge-low'
    return f'<div class="{cls}">⬤ {label}</div>'

def get_channel_class(channel_str):
    c = channel_str.lower()
    if 'call' in c: return 'channel-call'
    if 'email' in c: return 'channel-email'
    if 'sms' in c: return 'channel-sms'
    if 'app' in c: return 'channel-app'
    return 'channel-email' 

# SECTION 2 — Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>◈ RetainAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0;'>Agentic Churn Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        from backend.models.predict import MODEL_NAME, lr_f1, dt_f1
        f1_score = max(lr_f1, dt_f1)
    except:
        MODEL_NAME = "Logistic Regression"
        f1_score = 0.81
        
    st.markdown("### Model Information")
    st.markdown(f"**Best Model**: {MODEL_NAME}")
    st.markdown(f"**F1 Score**: {f1_score:.2f}")
    
    st.write("")
    st.markdown("### Top Churn Predictors")
    st.markdown("- Month-to-month Contract")
    st.markdown("- High Monthly/Total Charges")
    st.markdown("- Fiber Optic Internet")
    st.markdown("- Short Tenure")
    st.markdown("- Electronic Check Payment")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 13px; color: #a0a0a0;'>Powered by LangGraph + RAG</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Scenario Simulator"])

# Ensure session state initialization
keys_defaults = {
    "rtai_gender": "Female", "rtai_senior": "No", "rtai_partner": "No", "rtai_deps": "No",
    "rtai_phone": "Yes", "rtai_mult": "No", "rtai_internet": "Fiber optic",
    "rtai_sec": "No", "rtai_backup": "No", "rtai_protect": "No", "rtai_tech": "No",
    "rtai_tv": "No", "rtai_movies": "No",
    "rtai_tenure": 12, "rtai_contract": "Month-to-month", "rtai_paper": "Yes",
    "rtai_payment": "Electronic check", "rtai_monthly": 70.0, "rtai_total": 840.0
}
for k, v in keys_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def map_senior(val): return 1 if val == "Yes" else 0

# SECTION 3 — Tab 1
with tab1:
    col_input, col_res = st.columns([0.4, 0.6])
    
    with col_input:
        st.subheader("Customer Details")
        st.markdown("**Group 1: Demographics**")
        c1, c2 = st.columns(2)
        st.session_state["rtai_gender"] = c1.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(st.session_state["rtai_gender"]))
        st.session_state["rtai_senior"] = c2.selectbox("Senior Citizen", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["rtai_senior"]))
        c1, c2 = st.columns(2)
        st.session_state["rtai_partner"] = c1.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["rtai_partner"]))
        st.session_state["rtai_deps"] = c2.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["rtai_deps"]))
        
        st.markdown("**Group 2: Services**")
        st.session_state["rtai_internet"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(st.session_state["rtai_internet"]))
        
        c1, c2 = st.columns(2)
        st.session_state["rtai_phone"] = c1.selectbox("Phone Service", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["rtai_phone"]))
        st.session_state["rtai_mult"] = c2.selectbox("Multiple Lines", ["No", "No phone service", "Yes"], index=["No", "No phone service", "Yes"].index(st.session_state["rtai_mult"]))
        
        c1, c2 = st.columns(2)
        st.session_state["rtai_sec"] = c1.selectbox("Online Security", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_sec"]))
        st.session_state["rtai_backup"] = c2.selectbox("Online Backup", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_backup"]))
        
        c1, c2 = st.columns(2)
        st.session_state["rtai_protect"] = c1.selectbox("Device Protection", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_protect"]))
        st.session_state["rtai_tech"] = c2.selectbox("Tech Support", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_tech"]))
        
        c1, c2 = st.columns(2)
        st.session_state["rtai_tv"] = c1.selectbox("Streaming TV", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_tv"]))
        st.session_state["rtai_movies"] = c2.selectbox("Streaming Movies", ["No", "No internet service", "Yes"], index=["No", "No internet service", "Yes"].index(st.session_state["rtai_movies"]))
        
        st.markdown("**Group 3: Account**")
        st.session_state["rtai_tenure"] = st.slider("Tenure (Months)", 0, 72, st.session_state["rtai_tenure"])
        c1, c2 = st.columns(2)
        st.session_state["rtai_contract"] = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state["rtai_contract"]))
        st.session_state["rtai_paper"] = c2.selectbox("Paperless Billing", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["rtai_paper"]))
        st.session_state["rtai_payment"] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state["rtai_payment"]))
        
        c1, c2 = st.columns(2)
        st.session_state["rtai_monthly"] = float(c1.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=float(st.session_state["rtai_monthly"])))
        st.session_state["rtai_total"] = float(c2.number_input("Total Charges", min_value=0.0, max_value=12000.0, value=float(st.session_state["rtai_total"])))
        
        analyze_btn = st.button("◈ Analyze Customer")

    with col_res:
        if analyze_btn:
            customer_data = {
                "gender": st.session_state["rtai_gender"], "SeniorCitizen": map_senior(st.session_state["rtai_senior"]),
                "Partner": st.session_state["rtai_partner"], "Dependents": st.session_state["rtai_deps"],
                "tenure": st.session_state["rtai_tenure"], "PhoneService": st.session_state["rtai_phone"],
                "MultipleLines": st.session_state["rtai_mult"], "InternetService": st.session_state["rtai_internet"],
                "OnlineSecurity": st.session_state["rtai_sec"], "OnlineBackup": st.session_state["rtai_backup"],
                "DeviceProtection": st.session_state["rtai_protect"], "TechSupport": st.session_state["rtai_tech"],
                "StreamingTV": st.session_state["rtai_tv"], "StreamingMovies": st.session_state["rtai_movies"],
                "Contract": st.session_state["rtai_contract"], "PaperlessBilling": st.session_state["rtai_paper"],
                "PaymentMethod": st.session_state["rtai_payment"],
                "MonthlyCharges": st.session_state["rtai_monthly"], "TotalCharges": st.session_state["rtai_total"]
            }
            st.session_state['rtai_current_customer'] = customer_data
            
            with st.spinner("Running ML Models..."):
                st.session_state['rtai_churn_res'] = predict_churn(customer_data)
                
            with st.spinner("◈ Agent reasoning (RAG context loading)..."):
                st.session_state['rtai_plan'] = run_retention_agent(customer_data, st.session_state['rtai_churn_res'])
                
            st.session_state["rtai_feedback_given"] = False
            
        if 'rtai_churn_res' in st.session_state:
            customer_data = st.session_state['rtai_current_customer']
            churn_res = st.session_state['rtai_churn_res']
            plan = st.session_state.get('rtai_plan', {})
            
            prob = churn_res["churn_probability"]
            if prob < 0.35: risk_level = "Low"
            elif prob <= 0.60: risk_level = "Medium"
            elif prob <= 0.80: risk_level = "High"
            else: risk_level = "Critical"
            
            st.markdown(get_risk_html(risk_level), unsafe_allow_html=True)
            st.write("")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"<div class='metric-card'><h4>Churn Probability</h4><h2>{prob:.1%}</h2></div>", unsafe_allow_html=True)
            conf = max(prob, 1 - prob)
            mc2.markdown(f"<div class='metric-card'><h4>Confidence</h4><h2>{conf:.1%}</h2></div>", unsafe_allow_html=True)
            mc3.markdown(f"<div class='metric-card'><h4>Model Used</h4><h2>{churn_res['model_used']}</h2></div>", unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': ""},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 35], 'color': "#00C851"},
                        {'range': [35, 60], 'color': "#FFD700"},
                        {'range': [60, 80], 'color': "#FF8C00"},
                        {'range': [80, 100], 'color': "#FF4444"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("### Factors Driving This Prediction")
            shap_vals = churn_res.get("shap_values", {})
            if shap_vals:
                sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]))[-10:]
                features = [k.split("__")[-1].replace("_", " ") for k, v in sorted_shap]
                vals = [v for k, v in sorted_shap]
                colors = ['#FF4444' if v > 0 else '#00C851' for v in vals]
                
                fig_shap = go.Figure(go.Bar(
                    x=vals, y=features, orientation='h',
                    marker_color=colors
                ))
                fig_shap.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                st.plotly_chart(fig_shap, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### AI Retention Strategy")
            if plan and isinstance(plan, dict):
                st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; color:#495057;'><i>{plan.get('risk_analysis', '')}</i></div>", unsafe_allow_html=True)
                st.write("")
                
                st.markdown("**Churn Drivers**")
                drivers_html = ""
                for dr in plan.get("churn_drivers", []):
                    drivers_html += f"<span style='display:inline-block; background-color:#e9ecef; color:black; border-radius:15px; padding:4px 10px; margin-right:5px; margin-bottom:5px; font-size:13px;'>{dr}</span>"
                st.markdown(drivers_html, unsafe_allow_html=True)
                st.write("")
                
                st.markdown("**Recommended Actions**")
                for act in plan.get("recommended_actions", []):
                    channel = act.get('channel', 'Email')
                    c_style = get_channel_class(channel)
                    st.markdown(f"""
                    <div class='action-card'>
                        <b>{act.get('action', '')}</b><br/>
                        <span style='color: #6c757d; font-size: 14px;'>{act.get('rationale', '')}</span><br/>
                        <div style='margin-top: 8px;'>
                            <span class='{c_style}'>{channel}</span>
                            <span style='font-size: 13px; color: #495057;'>⏳ {act.get('timeline', '')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"<div class='impact-box'>Expected Impact: {plan.get('expected_impact', 'Unknown')}</div>", unsafe_allow_html=True)
                
                srcs = plan.get("sources", [])
                if srcs:
                    st.markdown(f"<span style='font-size:12px; color:#adb5bd;'>Sources: {', '.join(srcs)}</span>", unsafe_allow_html=True)
                    
                st.info(plan.get("ethical_disclaimer", "AI-generated plan. Human review recommended."))

                st.markdown("---")
                st.markdown("**Was this retention plan helpful?**")
                
                if "rtai_feedback_given" not in st.session_state:
                    st.session_state["rtai_feedback_given"] = False
                    
                if not st.session_state["rtai_feedback_given"]:
                    comment = st.text_input("Optional comment on this plan:", key="rtai_feedback_comment")
                    fcol1, fcol2, _ = st.columns([1, 1, 4])
                    
                    if fcol1.button("👍 Helpful", use_container_width=True):
                        from backend.feedback import save_feedback
                        c_profile = {
                            "tenure_months": customer_data["tenure"],
                            "contract_type": customer_data["Contract"],
                        }
                        save_feedback(c_profile, plan, rating=1, comment=comment)
                        st.session_state["rtai_feedback_given"] = True
                        st.rerun()

                    if fcol2.button("👎 Not helpful", use_container_width=True):
                        from backend.feedback import save_feedback
                        c_profile = {
                            "tenure_months": customer_data["tenure"],
                            "contract_type": customer_data["Contract"],
                        }
                        save_feedback(c_profile, plan, rating=0, comment=comment)
                        st.session_state["rtai_feedback_given"] = True
                        st.rerun()
                else:
                    st.success("Thank you for your feedback!")

# SECTION 4 — Tab 2: Batch Analysis
with tab2:
    st.subheader("Batch Analytics")
    uploaded_file = st.file_uploader("Upload Customer Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        if st.button("◈ Run Batch Analysis"):
            progress_bar = st.progress(0)
            results = []
            
            for i, row in df.iterrows():
                customer_data = row.to_dict()
                churn_res = predict_churn(customer_data)
                
                prob = churn_res["churn_probability"]
                if prob < 0.35: level = "Low"
                elif prob <= 0.60: level = "Medium"
                elif prob <= 0.80: level = "High"
                else: level = "Critical"
                
                top_driver = ""
                shap_vals = churn_res.get("shap_values", {})
                if shap_vals:
                    top_key = sorted(shap_vals.keys(), key=lambda k: abs(shap_vals[k]))[-1]
                    top_driver = top_key.split("__")[-1].replace("_", " ")
                
                res_row = {
                    "churn_probability": round(prob, 4),
                    "risk_level": level,
                    "top_driver": top_driver
                }
                if "customerID" in customer_data:
                    res_row["customerID"] = customer_data["customerID"]
                results.append(res_row)
                
                progress_bar.progress((i + 1) / len(df))
                
            res_df = pd.DataFrame(results)
            def code_risk(val):
                color = '#00C851' if val == 'Low' else '#FFD700' if val == 'Medium' else '#FF8C00' if val == 'High' else '#FF4444'
                return f'color: {color}; font-weight: bold'
                
            try:
                styled_df = res_df.style.map(code_risk, subset=['risk_level'])
            except:
                styled_df = res_df.style.applymap(code_risk, subset=['risk_level'])
                
            st.dataframe(styled_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", data=csv, file_name="batch_analysis_results.csv", mime="text/csv")
            
            st.caption("Agent-powered detailed plans available for individual customers in Tab 1")

# SECTION 5 — Tab 3: Scenario Simulator
with tab3:
    st.title("What-If Scenario Simulator")
    st.markdown("Adjust key factors to see how retention interventions change churn risk")
    
    scol1, scol2 = st.columns(2)
    with scol1:
        st.subheader("Intervention Controls")
        
        sim_contract = st.selectbox("Contract upgrade", ["Keep current", "Upgrade to 1 year", "Upgrade to 2 year"])
        sim_discount = st.slider("Discount applied (% reduction)", 0, 30, 0)
        sim_tech = st.checkbox("Add TechSupport")
        sim_sec = st.checkbox("Add OnlineSecurity")
        sim_autopay = st.checkbox("Switch to AutoPay (Credit card/Bank transfer)")
        
        st.write("")
        sim_btn = st.button("◈ Simulate Outcomes")
        
    if sim_btn:
        orig_cust = {
            "gender": st.session_state["rtai_gender"], "SeniorCitizen": map_senior(st.session_state["rtai_senior"]),
            "Partner": st.session_state["rtai_partner"], "Dependents": st.session_state["rtai_deps"],
            "tenure": st.session_state["rtai_tenure"], "PhoneService": st.session_state["rtai_phone"],
            "MultipleLines": st.session_state["rtai_mult"], "InternetService": st.session_state["rtai_internet"],
            "OnlineSecurity": st.session_state["rtai_sec"], "OnlineBackup": st.session_state["rtai_backup"],
            "DeviceProtection": st.session_state["rtai_protect"], "TechSupport": st.session_state["rtai_tech"],
            "StreamingTV": st.session_state["rtai_tv"], "StreamingMovies": st.session_state["rtai_movies"],
            "Contract": st.session_state["rtai_contract"], "PaperlessBilling": st.session_state["rtai_paper"],
            "PaymentMethod": st.session_state["rtai_payment"],
            "MonthlyCharges": st.session_state["rtai_monthly"], "TotalCharges": st.session_state["rtai_total"]
        }
        
        new_cust = orig_cust.copy()
        if sim_contract == "Upgrade to 1 year":
            new_cust["Contract"] = "One year"
        elif sim_contract == "Upgrade to 2 year":
            new_cust["Contract"] = "Two year"
            
        if sim_discount > 0:
            new_cust["MonthlyCharges"] = new_cust["MonthlyCharges"] * (1 - sim_discount/100.0)
            
        if sim_tech and new_cust["InternetService"] != "No":
            new_cust["TechSupport"] = "Yes"
            
        if sim_sec and new_cust["InternetService"] != "No":
            new_cust["OnlineSecurity"] = "Yes"
            
        if sim_autopay:
            new_cust["PaymentMethod"] = "Credit card (automatic)"
            
        orig_res = predict_churn(orig_cust)
        new_res = predict_churn(new_cust)
        
        o_prob = orig_res["churn_probability"]
        n_prob = new_res["churn_probability"]
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<h3 style='text-align: center'>Current Risk</h3>", unsafe_allow_html=True)
            o_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=o_prob*100, 
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 35], 'color': "#00C851"},
                        {'range': [35, 60], 'color': "#FFD700"},
                        {'range': [60, 80], 'color': "#FF8C00"},
                        {'range': [80, 100], 'color': "#FF4444"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': o_prob * 100}
                }
            ))
            o_gauge.update_layout(height=250, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(o_gauge, use_container_width=True)
            
        with c2:
            delta = (o_prob - n_prob) / o_prob * 100 if o_prob > 0 else 0
            color = "#00C851" if delta > 0 else "#FF4444"
            st.markdown(f"<div style='text-align: center; margin-top: 80px;'><h2 style='color: {color};'>Risk Reduction: {delta:.1f}%</h2></div>", unsafe_allow_html=True)
            
        with c3:
            st.markdown("<h3 style='text-align: center'>After Intervention</h3>", unsafe_allow_html=True)
            n_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=n_prob*100, 
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 35], 'color': "#00C851"},
                        {'range': [35, 60], 'color': "#FFD700"},
                        {'range': [60, 80], 'color': "#FF8C00"},
                        {'range': [80, 100], 'color': "#FF4444"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': n_prob * 100}
                }
            ))
            n_gauge.update_layout(height=250, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(n_gauge, use_container_width=True)

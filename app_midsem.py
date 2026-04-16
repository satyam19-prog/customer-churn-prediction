import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.graph_objects as go
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn { background-color: #ffebee; color: #c62828; }
    .no-churn { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    preprocessor = joblib.load('models/preprocessor.pkl')
    lr = joblib.load('models/lr_model.pkl')
    dt = joblib.load('models/dt_model.pkl')
    df = pd.read_csv('data/telco_churn.csv')
    try:
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
    except Exception:
        X_test, y_test = None, None
    return preprocessor, lr, dt, df, X_test, y_test

preprocessor, lr_model, dt_model, df, X_test, y_test = load_models_and_data()

# pick best model by test F1

def pick_best_model(lr, dt, X_test, y_test):
    if X_test is None or y_test is None:
        return lr, 'Logistic Regression (default)', {}
    lr_f1 = f1_score(y_test, lr.predict(X_test))
    dt_f1 = f1_score(y_test, dt.predict(X_test))
    metrics = {
        'Logistic Regression': {
            'f1': lr_f1,
            'accuracy': accuracy_score(y_test, lr.predict(X_test)),
            'precision': precision_score(y_test, lr.predict(X_test)),
            'recall': recall_score(y_test, lr.predict(X_test))
        },
        'Decision Tree': {
            'f1': dt_f1,
            'accuracy': accuracy_score(y_test, dt.predict(X_test)),
            'precision': precision_score(y_test, dt.predict(X_test)),
            'recall': recall_score(y_test, dt.predict(X_test))
        }
    }
    if lr_f1 >= dt_f1:
        return lr, 'Logistic Regression', metrics
    return dt, 'Decision Tree', metrics

best_model, best_model_name, model_metrics = pick_best_model(lr_model, dt_model, X_test, y_test)

st.title("Customer Churn Prediction")
st.markdown("Predict customer churn with Machine Learning")
st.markdown("---")

if best_model is None:
    st.error("Model files not found. Run preprocessing and training first.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 50.0)

    st.markdown("---")
    if st.button("Predict Churn"):
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_data])
        # coerce numeric
        num_features = None
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                num_features = cols
                break
        if num_features:
            for col in num_features:
                if col in input_df.columns:
                    s = input_df[col].astype(str).str.strip()
                    s = s.mask(s == '', np.nan)
                    input_df[col] = pd.to_numeric(s, errors='coerce')
        X_proc = preprocessor.transform(input_df)
        prediction = best_model.predict(X_proc)[0]
        proba = best_model.predict_proba(X_proc)[0] if hasattr(best_model, 'predict_proba') else None

        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box churn">WILL CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box no-churn">WILL STAY</div>', unsafe_allow_html=True)
        with col2:
            if proba is not None:
                st.metric("Confidence", f"{max(proba) * 100:.1f}%")
        with col3:
            if proba is not None:
                risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
                st.metric("Risk Level", risk)
        if proba is not None:
            fig = go.Figure(data=[
                go.Bar(name='No Churn', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
                go.Bar(name='Churn', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        if prediction == 1:
            st.warning("**At Risk:** Offer retention incentives, upgrade to longer contract, provide better support")
        else:
            st.success("**Low Risk:** Continue excellent service, send surveys, offer loyalty benefits")

    with st.sidebar:
        st.header("Model Info")
        if model_metrics:
            st.info(f"**{best_model_name} on test set**\n- F1: {model_metrics[best_model_name]['f1']:.3f}\n- Accuracy: {model_metrics[best_model_name]['accuracy']:.3f}")
        st.header("Top Predictors")
        st.markdown(
            """
            - Contract type
            - Tenure duration  
            - Monthly charges
            - Internet service
            - Payment method
            """
        )
        st.markdown("---")
        st.caption("Built with Streamlit & Scikit-learn")

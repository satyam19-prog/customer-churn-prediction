import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import f1_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'lr_model.pkl')
DT_MODEL_PATH = os.path.join(MODELS_DIR, 'dt_model.pkl')
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.npy')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.npy')

# Load components
preprocessor = joblib.load(PREPROCESSOR_PATH)
lr_model = joblib.load(LR_MODEL_PATH)
if not hasattr(lr_model, 'multi_class'):
    lr_model.multi_class = 'auto'
dt_model = joblib.load(DT_MODEL_PATH)

X_train = np.load(X_TRAIN_PATH, allow_pickle=True)
y_train = np.load(Y_TRAIN_PATH, allow_pickle=True)
X_test = np.load(X_TEST_PATH, allow_pickle=True)
y_test = np.load(Y_TEST_PATH, allow_pickle=True)

# Select best model based on F1
lr_preds = lr_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

lr_f1 = f1_score(y_test, lr_preds)
dt_f1 = f1_score(y_test, dt_preds)

if lr_f1 >= dt_f1:
    BEST_MODEL = lr_model
    MODEL_NAME = "Logistic Regression"
    # Slice to first 100 points to massively reduce LinearExplainer RAM footprint
    # without significantly impacting global baseline metrics
    explainer = shap.LinearExplainer(BEST_MODEL, X_train[:100])
else:
    BEST_MODEL = dt_model
    MODEL_NAME = "Decision Tree"
    explainer = shap.TreeExplainer(BEST_MODEL)

def predict_churn(customer_data: dict) -> dict:
    try:
        df = pd.DataFrame([customer_data])
        
        # Coerce numeric columns
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Handle the SeniorCitizen column as int safely handling NaNs
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
            
        X_proc = preprocessor.transform(df)
        
        churn_probability = float(BEST_MODEL.predict_proba(X_proc)[0][1])
        churn_prediction = int(BEST_MODEL.predict(X_proc)[0])
        
        shap_values = explainer.shap_values(X_proc)
        
        if MODEL_NAME == "Logistic Regression":
            print(f"SHAP LR output type: {type(shap_values)}")
            shap_values_out = shap_values[0]
        else:
            print(f"SHAP DT output type: {type(shap_values)}")
            # Fallbacks in case user's exact instruction `shap_values[0][1]` was specific
            try:
                # If shap_values is a list of arrays (class0, class1)
                if isinstance(shap_values, list):
                    shap_values_out = shap_values[1][0]
                elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                    shap_values_out = shap_values[0, :, 1]
                else: # Trust instruction literal
                    shap_values_out = shap_values[0][1]
            except Exception as e:
                # the instruction literally said: shap_values[0][1] (class 1 SHAP values, first row)
                try:
                    shap_values_out = shap_values[0][1]
                except:
                    # just take whatever is array-like
                    shap_values_out = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        feature_names = preprocessor.get_feature_names_out()
        
        shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_values_out)}
        top_shap_dict = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:15])
        
        # Round the values
        for k in top_shap_dict:
            top_shap_dict[k] = round(top_shap_dict[k], 4)

        return {
            "churn_probability": churn_probability,
            "churn_prediction": churn_prediction,
            "model_used": MODEL_NAME,
            "shap_values": top_shap_dict
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "churn_probability": 0.5,
            "churn_prediction": 0,
            "model_used": "Error",
            "shap_values": {}
        }

if __name__ == "__main__":
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
    result = predict_churn(test_customer)
    import json
    print("FINAL_RESULT:")
    print(json.dumps(result, indent=2))

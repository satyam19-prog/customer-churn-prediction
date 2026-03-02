## Customer Churn Prediction (Telco)

A small end-to-end machine learning project that predicts whether a telecom customer is likely to churn, using their demographic, service usage, and billing information.  
The best model is exposed through a simple Streamlit web app for interactive scoring.

---

## 1. Project Structure

- `data_prep.py` – builds the preprocessing pipeline (imputation, scaling, one-hot encoding) and saves:
  - `models/preprocessor.pkl`
  - `data/X_train.npy`, `data/X_test.npy`, `data/y_train.npy`, `data/y_test.npy`
- `model_training.py` – trains Logistic Regression and Decision Tree models, evaluates them, prints metrics, and saves:
  - `models/lr_model.pkl`, `models/dt_model.pkl`
  - plots under `plots/` (confusion matrix, decision tree)
- `app.py` – Streamlit app for entering customer details and viewing churn predictions.
- `report.tex` – full technical LaTeX report for this project.

---

## 2. Tech Stack

- **Language**: Python 3  
- **Core libraries**: `pandas`, `numpy`, `scikit-learn`, `joblib`  
- **Visualisation**: `matplotlib`, `seaborn`, `plotly`  
- **Web app**: `streamlit`  

Install everything via:

```bash
pip install -r requirements.txt
```

---

## 3. Getting Started

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd customer-churn-prediction
```

2. **Place the dataset**

Put the Telco churn CSV as:

```text
data/telco_churn.csv
```

3. **Create a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Running the Pipeline

1. **Preprocess data**

```bash
python3 data_prep.py
```

2. **Train and evaluate models**

```bash
python3 model_training.py
```

This prints metrics such as accuracy, precision, recall, F1/F2, R², MSE, MAE, and MASE for both models and saves plots under `plots/`.

3. **Launch the Streamlit app**

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) to interact with the churn prediction UI.

---

## 5. Interpreting the App

- Fill in customer demographics, subscribed services, and account details in the form.  
- Click **Predict Churn** to see:
  - predicted label: “will churn” / “will stay”,
  - model confidence (probability),
  - qualitative risk band (Low / Medium / High),
  - a small probability bar chart.
- The sidebar shows which model is currently used and summary metrics from the test set.

---

## 6. Notes and Extensions

- The LaTeX report in `report.tex` contains a detailed description of the methodology and results.  
- You can extend this project with more models, hyperparameter tuning, class-imbalance handling, or richer explanations (e.g. SHAP) as next steps.


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Load Data & Fix Trap
df = pd.read_csv('data/telco_churn.csv')
print("--- BEFORE FIX ---")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].dtypes)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\n--- AFTER FIX ---")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].dtypes)
print(f"\nExposed missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")




# 2. Separate Features and Target
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.select_dtypes(include=['object', 'str']).columns.tolist()

# 3. Build Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. SPLIT DATA FIRST (Prevents Data Leakage - HUGE for Viva Marks!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit & Transform Training Data, but ONLY Transform Test Data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n--- DATA LEAKAGE PREVENTED ---")
print(f"Training Data Shape: {X_train_processed.shape}")
print(f"Testing Data Shape: {X_test_processed.shape}")

joblib.dump(preprocessor, 'models/preprocessor.pkl')
np.save('data/X_train.npy', X_train_processed)
np.save('data/X_test.npy', X_test_processed)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("\nPipeline saved to 'models/preprocessor.pkl'")
print("Processed data saved to 'data/' folder. Ready for Member 2!")
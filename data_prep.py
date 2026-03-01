"""
Customer Churn Prediction - Data Preprocessing Pipeline
Handles missing values, encoding, scaling, and train-test splitting.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

print("Starting data preprocessing pipeline...")

# Load raw data and handle hidden blank spaces in TotalCharges
df = pd.read_csv('data/telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Define features and target variable
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.select_dtypes(include=['object', 'str']).columns.tolist()

# Construct transformation pipelines
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

# Split data to prevent data leakage prior to applying transformations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit on training data, transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Ensure output directories exist and serialize outputs
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

joblib.dump(preprocessor, 'models/preprocessor.pkl')
np.save('data/X_train.npy', X_train_processed)
np.save('data/X_test.npy', X_test_processed)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print(f"Pipeline complete. Training data shape: {X_train_processed.shape}")
print("Preprocessor and data arrays successfully saved.")
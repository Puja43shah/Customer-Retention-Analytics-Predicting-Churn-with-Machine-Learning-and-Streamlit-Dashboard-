# EDA_and_Modeling.py
"""
End-to-End Machine Learning Project (Customer Churn)
Steps:
1. Import and Describe Dataset
2. Perform EDA and interpret
3. Decide on Modeling Approach
4. Train & Evaluate with K-Fold + GridSearchCV
5. Save Best Model
6. Load and Predict for New Data
"""

# ---------------- LIBRARIES ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ---------------- CONFIG ----------------
DATA_PATH = r"Churn_Modelling.csv"  # Change if needed
MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler_info.joblib"
RANDOM_STATE = 42

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print(df.info())
print(df.describe().T)
print("Missing values per column:\n", df.isna().sum())

# ---------------- FEATURE ENGINEERING ----------------
# Add engineered features
df['CreditUtilization'] = df['Balance'] / (df['CreditScore'] + 1e-9)
df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-9)
df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']

# Credit Score Groups
bins = [0, 669, 739, 850]
labels = ['Low', 'Medium', 'High']
df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

# Encode categorical features
encoder = LabelEncoder()
for col in ['Geography', 'Gender', 'CreditScoreGroup']:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Prepare features and labels
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
X = df.drop(columns=drop_cols + ['Exited'], errors='ignore')
y = df['Exited'].astype(int)

# ---------------- SPLIT & SCALE ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=RANDOM_STATE, stratify=y)
scaling_columns = ['Age','CreditScore','Balance','EstimatedSalary','CreditUtilization',
                   'BalanceToSalaryRatio','CreditScoreAgeInteraction']

scaler = StandardScaler()
scaler.fit(X_train[scaling_columns])
X_train[scaling_columns] = scaler.transform(X_train[scaling_columns])
X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])

# ---------------- MODELING ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = 'f1'

# Models + hyperparameter grids
models = {
    "Logistic Regression": (LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE),
                            {"C": [0.1, 1.0, 10.0]}),

    "Random Forest": (RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
                      {"n_estimators": [100, 200],
                       "max_depth": [6, 10, None],
                       "min_samples_split": [2, 5]}),

    "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                          {"n_estimators": [100, 200],
                           "learning_rate": [0.05, 0.1]}),

    "Support Vector Machine": (ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('svc', SVC(probability=True, random_state=RANDOM_STATE))
    ]),
        {"svc__C": [0.5, 1.0],
         "svc__kernel": ['rbf', 'linear']}),

    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
                 {"n_estimators": [100, 200],
                  "max_depth": [3, 6],
                  "learning_rate": [0.05, 0.1]})
}

results = []
best_model = None
best_f1 = -1

# ---------------- GRID SEARCH ----------------
for name, (model, params) in models.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid = GridSearchCV(model, params, cv=skf, scoring=scoring, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_est.predict(X_test)
    if hasattr(best_est, "predict_proba"):
        y_proba = best_est.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"{name} - Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC_AUC: {roc}")
    results.append([name, acc, rec, f1, roc])

    if f1 > best_f1:
        best_f1 = f1
        best_model = best_est
        best_name = name

# ---------------- SAVE BEST MODEL ----------------
print(f"\nBest Model: {best_name} with F1 = {best_f1:.4f}")
joblib.dump({'model': best_model, 'scaler': scaler, 'scaler_columns': scaling_columns}, MODEL_PATH)
print("✅ Model saved as:", MODEL_PATH)

# ---------------- LOAD & PREDICT (DEMO) ----------------
loaded = joblib.load(MODEL_PATH)
loaded_model = loaded['model']
loaded_scaler = loaded['scaler']

# Predict a sample from test data
sample = X_test.iloc[[0]].copy()
sample[scaling_columns] = loaded_scaler.transform(sample[scaling_columns])
pred = loaded_model.predict(sample)[0]
print("\nSample prediction ->", "Churn" if pred == 1 else "No Churn")

# Save scaler separately for Streamlit app
joblib.dump({'scaler': scaler, 'scaler_columns': scaling_columns}, SCALER_PATH)
print("✅ Scaler info saved for Streamlit app.")

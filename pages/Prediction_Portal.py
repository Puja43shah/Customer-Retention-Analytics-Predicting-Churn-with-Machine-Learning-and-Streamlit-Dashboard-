# Predicting the churn
"""
Colorful Streamlit Dashboard for Customer Churn Prediction
----------------------------------------------------------
✨ Features:
- Custom colors and layout
- Animated gradient header
- Colored results and icons
- Supports single and batch prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction Portal", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        /* Background Gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #f6f9fc, #e9effd);
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #c7d2fe, #e0e7ff);
        }
        /* Header */
        .main-title {
            background: linear-gradient(to right, #4f46e5, #9333ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.3rem;
            font-weight: 900;
            text-align: center;
        }
        /* Form container */
        .stForm {
            background: #ffffffcc;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        /* Result box */
        .result-box {
            border-radius: 15px;
            padding: 15px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .positive {
            background: #dcfce7;
            color: #166534;
            border: 2px solid #22c55e;
        }
        .negative {
            background: #fee2e2;
            color: #991b1b;
            border: 2px solid #ef4444;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler_info.joblib"

@st.cache_resource
def load_model_and_scaler():
    model_info = joblib.load(MODEL_PATH)
    scaler_info = joblib.load(SCALER_PATH)
    return model_info['model'], scaler_info['scaler'], scaler_info['scaler_columns']

model, scaler, scaling_columns = load_model_and_scaler()

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-title">💼 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.write("Welcome! This dashboard predicts whether a customer is likely to **churn** or **stay loyal** based on their profile data.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Settings")
mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction (CSV Upload)"])

# ---------------- SINGLE PREDICTION ----------------
if mode == "Single Prediction":
    st.subheader("🎯 Predict for a Single Customer")

    with st.form("single_form"):
        col1, col2 = st.columns(2)
        geography = col1.selectbox("🌍 Geography (0=France, 1=Germany, 2=Spain)", [0, 1, 2])
        gender = col2.selectbox("👤 Gender (0=Female, 1=Male)", [0, 1])
        credit_score = col1.number_input("💳 Credit Score", min_value=300, max_value=850, value=650)
        age = col2.number_input("🎂 Age", min_value=18, max_value=100, value=35)
        tenure = col1.number_input("📅 Tenure (Years)", min_value=0, max_value=20, value=5)
        balance = col2.number_input("💰 Balance", min_value=0.0, value=60000.0, format="%.2f")
        num_products = col1.number_input("🛍️ Number of Products", min_value=1, max_value=10, value=2)
        has_cr_card = col2.selectbox("💳 Has Credit Card (0/1)", [0, 1])
        is_active = col1.selectbox("🔥 Is Active Member (0/1)", [0, 1])
        salary = col2.number_input("💵 Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

        submitted = st.form_submit_button("🔮 Predict Now")

    if submitted:
        # Engineered features
        df_input = pd.DataFrame([{
            "Geography": geography,
            "Gender": gender,
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
        }])

        df_input["CreditUtilization"] = df_input["Balance"] / (df_input["CreditScore"] + 1e-9)
        df_input["InteractionScore"] = df_input["NumOfProducts"] + df_input["HasCrCard"] + df_input["IsActiveMember"]
        df_input["BalanceToSalaryRatio"] = df_input["Balance"] / (df_input["EstimatedSalary"] + 1e-9)
        df_input["CreditScoreAgeInteraction"] = df_input["CreditScore"] * df_input["Age"]

        bins = [0, 669, 739, 850]
        df_input["CreditScoreGroup"] = pd.cut(df_input["CreditScore"], bins=bins, labels=[0, 1, 2], include_lowest=True).astype(int)

        df_input_scaled = df_input.copy()
        df_input_scaled[scaling_columns] = scaler.transform(df_input_scaled[scaling_columns])

        # Ensure correct feature order
        if hasattr(model, "feature_names_in_"):
            df_input_scaled = df_input_scaled[model.feature_names_in_]
        else:
            df_input_scaled = df_input_scaled.reindex(sorted(df_input_scaled.columns), axis=1)

        # Prediction
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_input_scaled)[:, 1][0]
        else:
            prob = None
        pred = model.predict(df_input_scaled)[0]

        # Result box
        if pred == 1:
            st.markdown(f'<div class="result-box negative">❌ The customer is **likely to CHURN**.<br>Churn Probability: {prob:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box positive">✅ The customer is **likely to STAY**.<br>Retention Probability: {(1 - prob):.2%}</div>', unsafe_allow_html=True)

# ---------------- BATCH PREDICTION ----------------
else:
    st.subheader("📊 Batch Prediction (Upload CSV)")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("Ensure your CSV includes: `Geography, Gender, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("📋 Uploaded data preview:")
        st.dataframe(df.head())

        # ✅ Encode categorical columns
        mapping_geo = {"France": 0, "Germany": 1, "Spain": 2}
        mapping_gender = {"Female": 0, "Male": 1}

        if "Geography" in df.columns:
            df["Geography"] = df["Geography"].map(mapping_geo)
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map(mapping_gender)
        

        # Add engineered features
        df["CreditUtilization"] = df["Balance"] / (df["CreditScore"] + 1e-9)
        df["InteractionScore"] = df["NumOfProducts"] + df["HasCrCard"] + df["IsActiveMember"]
        df["BalanceToSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1e-9)
        df["CreditScoreAgeInteraction"] = df["CreditScore"] * df["Age"]
        bins = [0, 669, 739, 850]
        df["CreditScoreGroup"] = pd.cut(df["CreditScore"], bins=bins, labels=[0, 1, 2], include_lowest=True).astype(int)

        # Scale numerical columns
        df_scaled = df.copy()
        df_scaled[scaling_columns] = scaler.transform(df_scaled[scaling_columns])

        # Ensure correct column order
        if hasattr(model, "feature_names_in_"):
            df_scaled = df_scaled[model.feature_names_in_]
        else:
            df_scaled = df_scaled.reindex(sorted(df_scaled.columns), axis=1)

        # Predict
        if hasattr(model, "predict_proba"):
            df["Predicted_Prob"] = model.predict_proba(df_scaled)[:, 1]
        df["Predicted_Churn"] = model.predict(df_scaled)

        st.success("✅ Predictions complete!")
        st.dataframe(df.head())

        # Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Predictions", csv, "churn_predictions.csv", "text/csv")

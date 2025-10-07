#Evaluating model performance

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("⚙️ Model Performance Dashboard")

# Load model info
model_info = joblib.load("best_model.joblib")
model = model_info['model']

st.subheader("Best Model Summary")
st.write(f"**Model Type:** {type(model).__name__}")

# Display feature importance if available
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else []
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

else:
    st.warning("Feature importance not available for this model type.")

# Optionally display a saved metrics CSV if you saved one during training
try:
    results = pd.read_csv("model_results.csv")
    st.subheader("Model Comparison Results")
    st.dataframe(results)
except:
    st.info("You can extend your training script to export model comparison results to `model_results.csv`.")

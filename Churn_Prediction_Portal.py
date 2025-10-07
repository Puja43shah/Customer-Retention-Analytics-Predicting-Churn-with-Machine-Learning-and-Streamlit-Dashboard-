# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="CUSTOMER CHURN DASHBOARD!",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; background: -webkit-linear-gradient(#4f46e5, #9333ea);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size:2.8rem;'>💼 Customer Churn Prediction Portal</h1>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to your **End-to-End Machine Learning Project Dashboard**!  
Use the sidebar to navigate through:
- 📊 **EDA Dashboard** — Visualize dataset insights  
- ⚙️ **Model Performance** — Compare algorithms and metrics  
- 🔮 **Prediction Portal** — Make single or batch predictions
""")

st.info("Navigate using the sidebar on the left to explore each section.")

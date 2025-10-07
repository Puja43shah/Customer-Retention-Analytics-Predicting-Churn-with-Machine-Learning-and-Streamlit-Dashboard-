#The front dashboard

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Exploratory Data Analysis (EDA)")

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Basic overview
st.subheader("Dataset Overview")
st.write(df.head())

# Column info
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Churn distribution
st.subheader("Target Variable Distribution (Exited)")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Exited', palette='coolwarm', ax=ax)
ax.set_title("Customer Churn Distribution")
st.pyplot(fig)

# Gender distribution
st.subheader("Gender Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Gender', hue='Exited', palette='viridis', ax=ax)
st.pyplot(fig)

# Geography vs churn
st.subheader("Churn by Geography")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Geography', hue='Exited', palette='crest', ax=ax)
st.pyplot(fig)

# Age distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Age'], kde=True, bins=20, color='teal')
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

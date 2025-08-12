import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import os

st.set_page_config(page_title="Training Data Overview", layout="wide")

# Load training data (go up one directory from 'pages')
base_path = os.path.dirname(__file__)  # current file's folder: /deployment/pages
model_path = os.path.join(base_path, "..", "model", "X_train.pkl")
X_train = joblib.load(os.path.abspath(model_path))

# Ensure it's a DataFrame
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)

# Map numeric labels to names
label_mapping = {0: "No Landslide", 1: "Landslide"}
X_train["label"] = X_train["label"].map(label_mapping)

# Display first few rows
st.markdown("## 📊 Sample of Training Data")
st.dataframe(X_train.head())

# Label Distribution
st.markdown("## Label Distribution")
label_counts = X_train['label'].value_counts().reset_index()
label_counts.columns = ["label", "count"]

fig_label = px.bar(
    label_counts,
    x="label",
    y="count",
    color="label",
    category_orders={"label": ["No Landslide", "Landslide"]},
    color_discrete_map={
        "No Landslide": "salmon",
        "Landslide": "teal"
    },
    labels={'label': 'Category', 'count': 'Number of Samples'},
    title="Landslide Distribution"
)
st.plotly_chart(fig_label, use_container_width=True)

# Feature summary
st.markdown("## 📈 Feature Summary")
st.write(X_train.describe())

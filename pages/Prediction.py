import streamlit as st
import requests
import numpy as np
import plotly.express as px
import io

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Prediction", page_icon="🤖", layout="wide")

API_URL = "http://127.0.0.1:8000/predict/"  # Change to your FastAPI endpoint

# ------------------------
# UI Title
# ------------------------
st.markdown("<h2 style='color: teal;'> Landslide Prediction</h2>", unsafe_allow_html=True)

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("📂 Upload your image file for prediction", type=["npy"])

if uploaded_file:
    # Send file to FastAPI
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    with st.spinner("Processing..."):
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                importance = result["interpretability"]

                # Display Prediction
                st.success(f"Prediction: **{prediction}**")

                # Feature Importance Plot
                importance_df = {f["feature"]: f["shap_value"] for f in importance}
                importance_sorted = sorted(importance_df.items(), key=lambda x: abs(x[1]), reverse=True)

                # Keep only top 10
                top_features = dict(importance_sorted[:10])

                fig = px.bar(
                    x=list(top_features.values()),  # SHAP values
                    y=[f"{i+1}. {feat}" for i, feat in enumerate(top_features.keys())],  # Numbered features
                    orientation='h',  # Horizontal bars = vertical stacking
                    labels={"x": "SHAP Value", "y": "Feature"},
                    title="🔍 Top 10 Feature Importance (SHAP Values)",
                    color=list(top_features.values()),
                    color_continuous_scale="RdBu"
                )

                # Make it big and readable
                fig.update_layout(
                    title_font=dict(size=28, color="teal"),
                    xaxis_title_font=dict(size=20),
                    yaxis_title_font=dict(size=20),
                    xaxis=dict(tickfont=dict(size=16)),
                    yaxis=dict(tickfont=dict(size=16)),
                    bargap=0.3,
                    height=700
                )

                st.plotly_chart(fig, use_container_width=True)






            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
else:
    st.info("Please upload an image to see predictions.")

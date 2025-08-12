import streamlit as st

# Page config
st.set_page_config(
    page_title="🌍 Landslide Detection",
    page_icon="🌋",
    layout="wide"
)

# Title & intro
st.markdown("<h1 style='text-align: center; color: teal;'>🌋 Landslide Detection App</h1>", unsafe_allow_html=True)
st.markdown("""
Welcome to the Landslide Detection tool.  
Upload your geospatial or sensor data to get predictions and insights.  
Use the sidebar to navigate between:
- **Prediction**: Get model predictions on your data  
- **Training Data**: Explore the dataset used to train the model
""")

# Image / hero section
st.image("assets/landslide_banner.jpg", use_container_width=True, caption="AI-powered Landslide Risk Prediction")



from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import shap
from data_prep import load_image, extract_additional_features, create_statistical_features_df

# -------------------
# FastAPI App
# -------------------
app = FastAPI()

# Allow CORS so Streamlit can access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features
model = joblib.load("model/lgb_model.pkl") 
features = joblib.load("model/selected_features.pkl")  # List of selected feature names

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read file bytes
    file_bytes = await file.read()
    
    # Data preparation
    img_array = load_image(file_bytes)
    img_array = np.expand_dims(img_array, axis=0)  # Ensure batch dimension
    img_array = extract_additional_features(img_array)
    features_df = create_statistical_features_df(img_array)
    final_df = features_df[features]  # Select only required features
    
    # Prediction
    prediction = model.predict(final_df)[0]
    result = "Landslide Detected" if prediction == 1 else "No Landslide"
    
    # SHAP for single prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(final_df)

    # LightGBM returns list for classification — pick class 1
    if isinstance(shap_values, list):
        shap_values_for_pred = shap_values[1][0]  # First record, class 1
    else:
        shap_values_for_pred = shap_values[0]  # Regression or binary

    # Create feature-importance list
    importance = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(features, shap_values_for_pred)
    ]

    return {
        "prediction": result,
        "interpretability": importance
    }

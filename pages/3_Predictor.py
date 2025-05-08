import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up paths - works both locally and in deployment
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # Create directory if it doesn't exist

def load_model():
    """Load the model with error handling"""
    try:
        # Try local file first
        local_path = os.path.join(MODEL_DIR, "rf_model.pkl")
        if os.path.exists(local_path):
            return joblib.load(local_path)
        
        # Fallback to GitHub raw URL
        model_url = "https://github.com/ajaykumarjaganathan/streamlit_telecom-customer_churn-detect/raw/main/models/rf_model.pkl"
        local_file, _ = urllib.request.urlretrieve(model_url)
        return joblib.load(local_file)
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.info(f"Current directory: {os.getcwd()}")
        st.info(f"Files in models directory: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
        return None

def main():
    st.title("Customer Churn Predictor")
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load prediction model. Please check if the model file exists.")
        return
    
    # Rest of your prediction code...
    # Add your input form and prediction logic here

if __name__ == "__main__":
    main()

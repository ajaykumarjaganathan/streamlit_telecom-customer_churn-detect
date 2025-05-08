import os
import urllib.request
import streamlit as st
import joblib
from pathlib import Path

def load_model():
    """Robust model loading with multiple fallback options"""
    # Define all possible model paths
    model_paths = [
        # 1. Local models directory
        Path(__file__).parent.parent / "models" / "rf_model.pkl",
        
        # 2. Absolute path (for some cloud deployments)
        Path("/mount/src/streamlit_telecom-customer_churn-detect/models/rf_model.pkl"),
        
        # 3. Relative path (alternative)
        Path("models/rf_model.pkl"),
    ]
    
    # GitHub fallback URL
    GITHUB_URL = "https://github.com/ajaykumarjaganathan/streamlit_telecom-customer_churn-detect/raw/main/models/rf_model.pkl"
    
    for path in model_paths:
        try:
            if path.exists():
                return joblib.load(path)
        except Exception as e:
            st.warning(f"Attempt failed for {path}: {str(e)}")
            continue
    
    # If all local attempts failed, try downloading from GitHub
    try:
        temp_file = "temp_model.pkl"
        urllib.request.urlretrieve(GITHUB_URL, temp_file)
        model = joblib.load(temp_file)
        st.success("Model downloaded from GitHub successfully!")
        return model
    except Exception as e:
        st.error(f"""
        ❌ Failed to load model from all sources:
        1. Local paths: {[str(p) for p in model_paths]}
        2. GitHub URL: {GITHUB_URL}
        
        Error: {str(e)}
        """)
        st.stop()

def main():
    st.title("Customer Churn Prediction")
    
    # Load model with progress indicator
    with st.spinner("Loading prediction model..."):
        try:
            model = load_model()
            st.success("✅ Model loaded successfully!")
            
            # Your prediction code here
            # Example:
            # prediction = model.predict(...)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()

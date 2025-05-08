# In your 3_Predictor.py, try this modified load_models function:
def load_models():
    try:
        # Load with explicit fix_imports and encoding parameters
        with open(preprocessor_path, 'rb') as file:
            preprocessor = joblib.load(file, fix_imports=True, encoding='latin1')
            
        with open(dt_model_path, 'rb') as file:
            dt_model = joblib.load(file, fix_imports=True, encoding='latin1')
            
        with open(rf_model_path, 'rb') as file:
            rf_model = joblib.load(file, fix_imports=True, encoding='latin1')
            
        return preprocessor, dt_model, rf_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None

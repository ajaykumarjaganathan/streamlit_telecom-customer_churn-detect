def load_models():
    try:
        with open("models/rf_model.pkl", "rb") as f:
            rf_model = joblib.load(f)
        st.success("Model loaded!")
        return rf_model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

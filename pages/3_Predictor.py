import streamlit as st
import joblib
import os
import pandas as pd

# Define model paths (adjust these to your actual file paths)
preprocessor_path = 'models/preprocessor.joblib'
dt_model_path = 'models/decision_tree_model.joblib'
rf_model_path = 'models/random_forest_model.joblib'

@st.cache_resource
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

def predict_churn(input_data, preprocessor, model):
    try:
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def main():
    # Load models once
    preprocessor, dt_model, rf_model = load_models()
    
    if None in [preprocessor, dt_model, rf_model]:
        st.error("Critical error: Could not load one or more models")
        return

    # UI Layout
    st.title("Predict Customer Churn")
    st.subheader("Please input the customer details below")

    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_revenue = st.number_input("Monthly Revenue", min_value=0.0, value=0.0, step=0.01)
        monthly_minutes = st.number_input("Monthly Minutes", min_value=0.0, value=0.0, step=0.01)
        director_calls = st.number_input("Director Assisted Calls", min_value=0, value=0, step=1)
        overage_minutes = st.number_input("Overage Minutes", min_value=0.0, value=0.0, step=0.01)
        roaming_calls = st.number_input("Roaming Calls", min_value=0, value=0, step=1)

    with col2:
        pct_change_minutes = st.number_input("Percentage Change Minutes", min_value=0.0, value=0.0, step=0.01)
        pct_change_revenues = st.number_input("Percentage Change Revenues", min_value=0.0, value=0.0, step=0.01)
        care_calls = st.number_input("Customer Care Calls", min_value=0, value=0, step=1)
        received_calls = st.number_input("Received Calls", min_value=0, value=0, step=1)
        outbound_calls = st.number_input("Outbound Calls", min_value=0, value=0, step=1)

    with col3:
        months_service = st.number_input("Months in Service", min_value=0, value=0, step=1)
        unique_subs = st.number_input("Unique Subs", min_value=0, value=0, step=1)
        active_subs = st.number_input("Active Subs", min_value=0, value=0, step=1)
        income_group = st.selectbox("Income Group", ["Low", "Medium", "High"])
        credit_rating = st.selectbox("Credit Rating", ["Poor", "Fair", "Good", "Excellent"])

    # Product selection (outside columns for full width)
    product = st.selectbox("Product", ["Basic", "Standard", "Premium"])

    # Model selection
    model_choice = st.radio("Select Model", ["Decision Tree", "Random Forest"])

    # Prediction button
    if st.button("Predict Churn"):
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([{
            'Monthly Revenue': monthly_revenue,
            'Monthly Minutes': monthly_minutes,
            'Director Assisted Calls': director_calls,
            'Overage Minutes': overage_minutes,
            'Roaming Calls': roaming_calls,
            'Percentage Change Minutes': pct_change_minutes,
            'Percentage Change Revenues': pct_change_revenues,
            'Customer Care Calls': care_calls,
            'Received Calls': received_calls,
            'Outbound Calls': outbound_calls,
            'Months in Service': months_service,
            'Unique Subs': unique_subs,
            'Active Subs': active_subs,
            'Income Group': income_group,
            'Credit Rating': credit_rating,
            'Product': product
        }])

        # Select model
        model = dt_model if model_choice == "Decision Tree" else rf_model

        # Make prediction
        prediction, probabilities = predict_churn(input_data, preprocessor, model)
        
        if prediction is not None:
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error("Prediction: Customer will churn")
            else:
                st.success("Prediction: Customer will stay")
            
            st.write(f"Confidence: {probabilities[prediction]*100:.2f}%")
            
            # Show probability breakdown
            st.write("Probability Breakdown:")
            st.write(f"- Will stay: {probabilities[0]*100:.2f}%")
            st.write(f"- Will churn: {probabilities[1]*100:.2f}%")

if __name__ == "__main__":
    main()

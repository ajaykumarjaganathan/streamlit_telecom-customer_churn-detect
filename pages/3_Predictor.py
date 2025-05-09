import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load model and preprocessing pipeline
@st.cache_resource
def load_artifacts():
    try:
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model files not found. Please ensure both 'churn_model.pkl' and 'preprocessor.pkl' exist.")
        return None, None

# Create sample data with consistent lengths
def create_sample_data():
    size = 1000
    return {
        'gender': np.random.choice(['Male', 'Female'], size),
        'SeniorCitizen': np.random.choice([0, 1], size),
        'Partner': np.random.choice(['Yes', 'No'], size),
        'Dependents': np.random.choice(['Yes', 'No'], size),
        'tenure': np.random.randint(1, 73, size),
        'PhoneService': np.random.choice(['Yes', 'No'], size),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size),
        'PaymentMethod': np.random.choice([
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ], size),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, size), 2),
        'TotalCharges': np.round(np.random.uniform(20, 8000, size), 2)
    }

def main():
    st.title("📱 Telecom Churn Prediction")
    st.markdown("Predict customer churn probability based on service details")

    model, preprocessor = load_artifacts()
    
    if model is None or preprocessor is None:
        if st.button("Create Demo Model"):
            # Create and save demo artifacts
            df = pd.DataFrame(create_sample_data())
            X = df
            y = np.random.choice(['Yes', 'No'], len(df))
            
            # Create preprocessing pipeline
            categorical_features = X.select_dtypes(include=['object']).columns
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
            
            # Fit preprocessing
            X_processed = preprocessor.fit_transform(X)
            
            # Train simple model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_processed, y)
            
            # Save artifacts
            with open('churn_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('preprocessor.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)
            
            st.success("Demo model created! Refresh the page to use it.")
        return

    # Input form
    with st.form("customer_input"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Details")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            
        with col2:
            st.subheader("Service Details")
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            
        col3, col4 = st.columns(2)
        
        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Preprocess input
            processed_input = preprocessor.transform(input_df)
            
            # Make prediction
            proba = model.predict_proba(processed_input)[0][1]
            
            # Display results
            st.subheader("Prediction Result")
            
            if proba > 0.7:
                st.error(f"🚨 High churn risk: {proba:.1%}")
                st.write("**Recommended actions:** Offer retention discount, assign account manager")
            elif proba > 0.4:
                st.warning(f"⚠️ Medium churn risk: {proba:.1%}")
                st.write("**Recommended actions:** Proactive service check, loyalty benefits")
            else:
                st.success(f"✅ Low churn risk: {proba:.1%}")
                st.write("**Status:** Customer appears satisfied")
            
            # Visual gauge
            st.progress(proba)
            st.caption(f"Churn probability: {proba:.1%}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check your input values and try again")

if __name__ == "__main__":
    main()

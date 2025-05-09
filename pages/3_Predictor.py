import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# Create and save demo model with proper encoding
def create_demo_model():
    # Create sample data
    np.random.seed(42)
    size = 1000
    data = {
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
        'TotalCharges': np.round(np.random.uniform(20, 8000, size), 2),
        'Churn': np.random.choice(['Yes', 'No'], size, p=[0.3, 0.7])
    }
    df = pd.DataFrame(data)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Save the pipeline
    with open('churn_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

# Load or create model
@st.cache_resource
def load_model():
    if not os.path.exists('churn_pipeline.pkl'):
        st.warning("Creating demo model...")
        return create_demo_model()
    
    try:
        with open('churn_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("📱 Telecom Churn Prediction")
    st.markdown("Predict customer churn probability based on service details")
    
    # Load model (will create demo if needed)
    model = load_model()
    
    if model is None:
        st.error("Failed to load or create model. Please check the error message.")
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
            # Make prediction (pipeline handles all preprocessing)
            proba = model.predict_proba(input_df)[0][1]
            
            # Display results
            st.subheader("Prediction Result")
            
            if proba > 0.7:
                st.error(f"🚨 High churn risk: {proba:.1%}")
                st.markdown("""
                **Recommended actions:**
                - Offer personalized retention discount
                - Assign dedicated account manager
                - Conduct exit interview
                """)
            elif proba > 0.4:
                st.warning(f"⚠️ Medium churn risk: {proba:.1%}")
                st.markdown("""
                **Recommended actions:**
                - Proactive service check
                - Offer loyalty benefits
                - Survey customer satisfaction
                """)
            else:
                st.success(f"✅ Low churn risk: {proba:.1%}")
                st.markdown("""
                **Status:** Customer appears satisfied
                **Suggestions:**
                - Continue current service quality
                - Consider upselling opportunities
                """)
            
            # Visual gauge
            st.progress(proba)
            st.metric("Churn Probability", f"{proba:.1%}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check your input values and try again")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Telecom Churn Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample data generation (replace with your actual data)
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    size = 1000
    data = {
        'customerID': [f'CUST{i:04d}' for i in range(size)],
        'gender': np.random.choice(['Male', 'Female'], size),
        'SeniorCitizen': np.random.choice([0, 1], size, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], size),
        'Dependents': np.random.choice(['Yes', 'No'], size),
        'tenure': np.random.randint(1, 72, size),
        'PhoneService': np.random.choice(['Yes', 'No'], size, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size, p=[0.5, 0.4, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size, p=[0.4, 0.4, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.3, 0.5, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.3, 0.5, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.4, 0.4, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size, p=[0.4, 0.4, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size, p=[0.6, 0.2, 0.2]),
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
    return pd.DataFrame(data)

# Load or create model
@st.cache_resource
def get_model():
    if os.path.exists('churn_model.pkl'):
        with open('churn_model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        # Train a simple model if none exists
        df = load_sample_data()
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = df['Churn']
        
        # Simple encoding
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col])
            
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model for future use
        with open('churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return model

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Predictor", "Data Explorer"])
    
    df = load_sample_data()
    model = get_model()
    
    if page == "Dashboard":
        st.title("📊 Telecom Churn Dashboard")
        
        # KPI metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            churn_rate = df[df['Churn'] == 'Yes'].shape[0] / len(df)
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with col3:
            avg_tenure = df['tenure'].mean()
            st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
        
        # Charts
        st.subheader("Churn Distribution")
        fig1 = px.pie(df, names='Churn', title='Churn Distribution')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Churn by Contract Type")
        fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Monthly Charges vs Tenure")
        fig3 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn')
        st.plotly_chart(fig3, use_container_width=True)
        
    elif page == "Predictor":
        st.title("🔮 Churn Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Details")
            gender = st.selectbox("Gender", ['Male', 'Female'])
            senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
            partner = st.selectbox("Partner", ['No', 'Yes'])
            dependents = st.selectbox("Dependents", ['No', 'Yes'])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            
        with col2:
            st.subheader("Service Details")
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
            tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
            payment_method = st.selectbox("Payment Method", [
                'Electronic check', 
                'Mailed check', 
                'Bank transfer (automatic)', 
                'Credit card (automatic)'
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
            
        if st.button("Predict Churn"):
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': 'Yes',  # simplified for demo
                'MultipleLines': 'No',
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': tech_support,
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': contract,
                'PaperlessBilling': 'Yes',
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': monthly_charges * tenure
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables (simplified)
            categorical_cols = input_df.select_dtypes(include=['object']).columns
            le = LabelEncoder()
            for col in categorical_cols:
                input_df[col] = le.fit_transform(input_df[col])
            
            # Make prediction
            proba = model.predict_proba(input_df)[0][1]
            
            # Display result
            st.subheader("Prediction Result")
            if proba > 0.5:
                st.error(f"High churn risk: {proba:.1%} probability")
                st.write("Recommended actions: Offer discount, improve service, or provide retention offer")
            else:
                st.success(f"Low churn risk: {proba:.1%} probability")
                st.write("Customer appears satisfied with current service")
                
            # Show probability gauge
            fig = px.indicator(
                mode="gauge+number",
                value=proba,
                title="Churn Probability",
                gauge={'axis': {'range': [0, 1]},
                       'steps': [
                           {'range': [0, 0.5], 'color': "lightgreen"},
                           {'range': [0.5, 1], 'color': "red"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Data Explorer":
        st.title("🔍 Data Explorer")
        
        st.write("Sample customer data (first 100 rows)")
        st.dataframe(df.head(100))
        
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            churn_filter = st.selectbox("Filter by Churn", ['All', 'Yes', 'No'])
        with col2:
            contract_filter = st.selectbox("Filter by Contract", ['All', 'Month-to-month', 'One year', 'Two year'])
            
        filtered_df = df.copy()
        if churn_filter != 'All':
            filtered_df = filtered_df[filtered_df['Churn'] == churn_filter]
        if contract_filter != 'All':
            filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]
            
        st.write(f"Showing {len(filtered_df)} records")
        st.dataframe(filtered_df)
        
        st.subheader("Summary Statistics")
        st.write(filtered_df.describe())

if __name__ == "__main__":
    main()

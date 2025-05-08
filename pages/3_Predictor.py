import streamlit as st
import pandas as pd
import os
import joblib
import pickle
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# Define file paths for models and preprocessor
models_dir = 'models'
dt_model_path = os.path.join(models_dir, 'dt_model.pkl')
rf_model_path = os.path.join(models_dir, 'rf_model.pkl')
preprocessor_path = os.path.join(models_dir, 'pipeline_preprocessor.pkl')

# Load models and preprocessor
@st.cache_resource
def load_models():
    with open(preprocessor_path, 'rb') as file:
        preprocessor = joblib.load(file)
    
    with open(dt_model_path, 'rb') as file:
        dt_model = joblib.load(file)
    
    with open(rf_model_path, 'rb') as file:
        rf_model = joblib.load(file)
    
    return preprocessor, dt_model, rf_model

preprocessor, dt_model, rf_model = load_models()

def predict_batch():
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Drop customerID column if exists
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle NaN values
        imputer = SimpleImputer(strategy='most_frequent')
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Convert numerical columns to appropriate dtype
        if 'TotalCharges' in df_filled.columns:
            df_filled['TotalCharges'] = pd.to_numeric(df_filled['TotalCharges'], errors='coerce')
        
        # Check if 'Churn' column exists
        if 'Churn' in df_filled.columns:
            # Convert 'Churn' to boolean if necessary
            if df_filled['Churn'].dtype == 'object':
                df_filled['Churn'] = df_filled['Churn'].map({'Yes': True, 'No': False})
            
            # Convert categorical variables to one-hot encoding
            categorical_cols = df_filled.select_dtypes(include=['object']).columns
            df_encoded = pd.get_dummies(df_filled, columns=categorical_cols, drop_first=True)
            
            # Separate features and target variable
            X = df_encoded.drop(columns=['Churn'])
            y = df_encoded['Churn']
            
            # Separate numerical and categorical columns
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

            # Standardize numerical features
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

            # Model selection
            model_choice = st.radio("Choose a model", ("SVM", "XGBoost"))

            if st.button("Predict"):
                if model_choice == "SVM":
                    model = SVC(random_state=42, probability=True)
                else:
                    model = XGBClassifier(random_state=42)
                
                model.fit(X, y)
                predictions = model.predict(X)
                
                # Calculate churn percentage
                churn_percentage = (predictions.sum() / len(predictions)) * 100
                display_results(churn_percentage, model_choice)
        else:
            st.error("Churn column not found in the dataset.")

def predict_online():
    st.header("Online Prediction")
    st.image("images/image.png", use_column_width=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Demographics')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

    with col2:
        st.subheader('Services')
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

    with col3:
        st.subheader('Payments')
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input('Monthly Charges', min_value=0)
        total_charges = st.number_input('Total Charges', min_value=0)
        tenure = st.number_input('Tenure', min_value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Preprocess data
        preprocessed_data = preprocessor.transform(input_data)

        selected_model = st.session_state.get('model', 'DecisionTree')
        model = dt_model if selected_model == 'DecisionTree' else rf_model

        prediction = model.predict_proba(preprocessed_data)
        churn_percentage = prediction[0][1] * 100
        display_results(churn_percentage, selected_model)

def display_results(churn_percentage, model_name):
    st.success(f'Churn Percentage ({model_name} Model): {churn_percentage:.2f}%')
    
    # Visualize churn risk
    st.subheader("Churn Risk Meter")
    colors = ['#8A2BE2', '#FFFF00', '#FFA500']  # Violet, Yellow, Orange
    thresholds = [20, 40]
    risk_level = np.digitize(churn_percentage, thresholds, right=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_percentage,
        title={'text': "Churn Risk"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': colors[risk_level]},
            'steps': [
                {'range': [0, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                {'range': [thresholds[1], 100], 'color': colors[2]}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Telecom Churn Prediction App", layout="wide")
    st.title("Customer Churn Prediction Application")
    
    prediction_option = st.radio(
        "Select Prediction Mode",
        ["Online Prediction", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_option == "Online Prediction":
        st.session_state['model'] = st.selectbox(
            'Select Model', 
            ['DecisionTree', 'RandomForest'],
            key='model_select'
        )
        predict_online()
    else:
        predict_batch()

if __name__ == '__main__':
    main()
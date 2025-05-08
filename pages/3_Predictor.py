import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('churn_dataset.csv')

# Data Preprocessing
def preprocess_data(df):
    # Drop customerID as it's not useful for prediction
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert TotalCharges to numeric, handling empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encoding categorical variables
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                     'Churn', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['InternetService'] = df['InternetService'].map({'DSL': 2, 'Fiber optic': 1, 'No': 0})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0, 
        'Mailed check': 1, 
        'Bank transfer (automatic)': 2, 
        'Credit card (automatic)': 3
    })
    
    return df

df = preprocess_data(df)

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save the model and scaler
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Streamlit App
def main():
    st.title('Customer Churn Prediction')
    st.subheader('Enter Customer Details')
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.radio('Gender', ['Male', 'Female'])
        senior_citizen = st.radio('Senior Citizen', ['Yes', 'No'])
        partner = st.radio('Partner', ['Yes', 'No'])
        dependents = st.radio('Dependents', ['Yes', 'No'])
        phone_service = st.radio('Phone Service', ['Yes', 'No'])
        multiple_lines = st.radio('Multiple Lines', ['Yes', 'No'])
        
    with col2:
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.radio('Online Security', ['Yes', 'No'])
        online_backup = st.radio('Online Backup', ['Yes', 'No'])
        device_protection = st.radio('Device Protection', ['Yes', 'No'])
        tech_support = st.radio('Tech Support', ['Yes', 'No'])
    
    st.markdown('---')
    
    col3, col4 = st.columns(2)
    
    with col3:
        streaming_tv = st.radio('Streaming TV', ['Yes', 'No'])
        streaming_movies = st.radio('Streaming Movies', ['Yes', 'No'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        
    with col4:
        paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', [
            'Electronic check', 
            'Mailed check', 
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ])
        tenure = st.slider('Tenure (months)', 1, 72, 12)
        monthly_charges = st.slider('Monthly Charges', 18.0, 120.0, 64.0)
        total_charges = st.slider('Total Charges', 18.0, 9000.0, 2000.0)
    
    # Convert inputs to model format
    input_data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'InternetService': 2 if internet_service == 'DSL' else 1 if internet_service == 'Fiber optic' else 0,
        'OnlineSecurity': 1 if online_security == 'Yes' else 0,
        'OnlineBackup': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection': 1 if device_protection == 'Yes' else 0,
        'TechSupport': 1 if tech_support == 'Yes' else 0,
        'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
        'Contract': 0 if contract == 'Month-to-month' else 1 if contract == 'One year' else 2,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'PaymentMethod': 0 if payment_method == 'Electronic check' else 1 if payment_method == 'Mailed check' else 2 if payment_method == 'Bank transfer (automatic)' else 3,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    if st.button('Predict Churn'):
        # Load model and scaler
        try:
            with open('churn_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            st.markdown('---')
            st.subheader('Prediction Result')
            
            if prediction[0] == 1:
                st.error(f'Churn Risk: High ({probability:.1%} probability)')
                st.write('This customer is likely to churn. Consider retention strategies.')
            else:
                st.success(f'Churn Risk: Low ({1-probability:.1%} probability)')
                st.write('This customer is likely to stay.')
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    main()

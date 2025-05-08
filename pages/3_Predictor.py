import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data (for the interface only)
df = pd.read_csv('churn_dataset.csv')
df.drop('customerID', axis=1, inplace=True)

# Create Encoding Dictionaries
yes_no_encoding = {'Yes': 1, 'No': 0}
gender_encoding = {'Male': 1, 'Female': 0}
internet_service_encoding = {'DSL': 2, 'Fiber optic': 1, 'None': 0}
contract_encoding = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_method_encoding = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}

# Streamlit Interface
st.title('Customer Churn Prediction')
st.subheader('Input Features')

# Collect user input
gender, senior_citizen, partner, dependents = st.columns(4)

with gender:
    gender = st.radio('Gender', ['Male', 'Female'])

with senior_citizen:
    senior_citizen = st.radio('Senior Citizen', ['Yes', 'No'])

with partner:
    partner = st.radio('Partner', ['Yes', 'No'])

with dependents:
    dependents = st.radio('Dependents', ['Yes', 'No'])

st.markdown('***')

phone_service, multiple_lines, online_security, online_backup = st.columns(4)

with phone_service:
    phone_service = st.radio('Phone Service', ['Yes', 'No'])

with multiple_lines:
    multiple_lines = st.radio('Multiple Lines', ['Yes', 'No'])

with online_security:
    online_security = st.radio('Online Security', ['Yes', 'No'])

with online_backup:
    online_backup = st.radio('Online Backup', ['Yes', 'No'])

st.markdown('***')

device_protection, tech_support, streaming_tv, streaming_movies, paperless_billing = st.columns(5)

with device_protection:
    device_protection = st.radio('Device Protection', ['Yes', 'No'])

with tech_support:
    tech_support = st.radio('Tech Support', ['Yes', 'No'])

with streaming_tv:
    streaming_tv = st.radio('Streaming TV', ['Yes', 'No'])

with streaming_movies:
    streaming_movies = st.radio('Streaming Movies', ['Yes', 'No'])

with paperless_billing:
    paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])

st.markdown('***')

internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'None'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox('Payment Method', [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
])
monthly_charges = st.slider('Monthly Charges', 
                           min_value=float(df.MonthlyCharges.min()),
                           max_value=float(df.MonthlyCharges.max()),
                           value=float(df.MonthlyCharges.mean()))

st.markdown('***')
st.subheader('Model Prediction')

if st.button('Predict'):
    # Show the unavailable message
    st.warning("Model prediction functionality is currently unavailable as the model file was removed.")
    
    # Optional: Show what the input would look like (for debugging/demo)
    st.write("Here's what your input would look like to the model:")
    
    # Convert inputs to model format (just for display, not actual prediction)
    input_features = {
        'gender': gender_encoding[gender],
        'SeniorCitizen': yes_no_encoding[senior_citizen],
        'Partner': yes_no_encoding[partner],
        'Dependents': yes_no_encoding[dependents],
        'PhoneService': yes_no_encoding[phone_service],
        'MultipleLines': yes_no_encoding[multiple_lines],
        'InternetService': internet_service_encoding[internet_service],
        'OnlineSecurity': yes_no_encoding[online_security],
        'OnlineBackup': yes_no_encoding[online_backup],
        'DeviceProtection': yes_no_encoding[device_protection],
        'TechSupport': yes_no_encoding[tech_support],
        'StreamingTV': yes_no_encoding[streaming_tv],
        'StreamingMovies': yes_no_encoding[streaming_movies],
        'Contract': contract_encoding[contract],
        'PaperlessBilling': yes_no_encoding[paperless_billing],
        'PaymentMethod': payment_method_encoding[payment_method],
        'MonthlyCharges': monthly_charges
    }
    
    # Display the formatted input
    st.json(input_features)
    
    # Placeholder for what the prediction would look like
    st.write("If the model were available, it would process these inputs and return a churn prediction.")

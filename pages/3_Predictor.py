import pandas as pd
import warnings
import time
import streamlit as st
import plotly.graph_objects as go
import PIL
from PIL import Image


df = pd.read_csv('churn_dataset.csv')

df.drop('customerID',axis = 1, inplace = True)
    
interface = st.container()

with interface:
    
    # Create Encoding Dictionaries
    yes_no_encoding = {'Yes': 1, 'No': 0}
    gender_encoding = {'Male': 1, 'Female': 0}
    internet_service_encoding = {'DSL': 2, 'Fiber optic': 1, 'None': 0}
    contract_encoding = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_method_encoding = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

    # Preprocess categorical columns in the DataFrame
    yes_no_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'MultipleLines',
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in yes_no_columns:
        df[col] = df[col].replace(yes_no_encoding)

    df['gender'] = df['gender'].replace(gender_encoding)
    df['InternetService'] = df['InternetService'].replace(internet_service_encoding)
    df['Contract'] = df['Contract'].replace(contract_encoding)
    df['PaymentMethod'] = df['PaymentMethod'].replace(payment_method_encoding)


    st.title('Enter details')
    st.subheader('Input Features')

    # Collect user input using Streamlit widgets
    gender,senior_citizen,partner,dependents = st.columns(spec = [1,1,1,1])

    with gender:
        
        gender = st.radio(label = 'Gender',options = ['Male','Female'])

    with senior_citizen:

        senior_citizen = st.radio(label = 'Are you senior citizen?',options = ['Yes','No'])

    with partner:

        partner = st.radio(label = 'Dou you have a partner?',options = ['Yes','No'])

    with dependents:

        dependents = st.radio(label = 'Do you have dependents?',options = ['Yes','No'])

    st.markdown(body = '***')


    phone_service, multiplelines, online_security,online_backup = st.columns(spec = [1,1,1,1])

    with phone_service:

        phone_service = st.radio(label = 'Phone Service',options = ['Yes','No'])

    with multiplelines:

        multiplelines = st.radio(label = 'MultipleLines',options = ['Yes','No'])

    with online_security:

        online_security = st.radio(label = 'Online Security',options = ['Yes','No'])

    with online_backup:

        online_backup = st.radio(label = 'Online backup',options = ['Yes','No'])


    st.markdown(body = '***')


    device_protection, tech_support, streaming_tv,streaming_movies,paperless_billing = st.columns(spec = [1,1,1,1,1])

    with device_protection:

        device_protection = st.radio(label = 'Device Protection',options = ['Yes','No'])

    with tech_support:

        tech_support = st.radio(label = 'Tech support',options = ['Yes','No'])

    with streaming_tv:

        streaming_tv = st.radio(label = 'Streaming TV',options = ['Yes','No'])

    with streaming_movies:

        streaming_movies = st.radio(label = 'Streaming Movies',options = ['Yes','No'])

    with paperless_billing:

        paperless_billing = st.radio(label = 'Paperless Billing',options = ['Yes','No'])


    st.markdown(body = '***')

    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'None'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check',
                                                         'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.slider('Monthly Charges', min_value=float(df.MonthlyCharges.min()),
                                    max_value=float(df.MonthlyCharges.max()), value=float(df.MonthlyCharges.mean()))

    # Convert categorical inputs to numerical using the encoding dictionaries
    
    gender = gender_encoding[gender]
    senior_citizen = yes_no_encoding[senior_citizen]
    partner = yes_no_encoding[partner]
    dependents = yes_no_encoding[dependents]
    phone_service = yes_no_encoding[phone_service]
    multiplelines = yes_no_encoding[multiplelines]
    online_security = yes_no_encoding[online_security]
    online_backup = yes_no_encoding[online_backup]
    device_protection = yes_no_encoding[device_protection]
    tech_support = yes_no_encoding[tech_support]
    streaming_tv = yes_no_encoding[streaming_tv]
    streaming_movies = yes_no_encoding[streaming_movies]
    paperless_billing = yes_no_encoding[paperless_billing]
    internet_service = internet_service_encoding[internet_service]
    contract = contract_encoding[contract]
    payment_method = payment_method_encoding[payment_method]

    # Create input_features DataFrame for model input
    input_features = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phone_service],
        'MultipleLines': [multiplelines],
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
        'MonthlyCharges': [monthly_charges]
    })

    
    st.markdown('***')
    
    st.subheader('Model Prediction')

    if st.button('Predict'):
        # Since we removed the model, we'll just show a placeholder message
        st.warning("Model prediction functionality is currently unavailable as the model file was removed.")
        
        # Placeholder for what would normally be the prediction
        # churn_probability = xgb_pipe.predict_proba(input_features)[0, 1]
        # st.markdown(f'Churn probability is ***{churn_probability:.0%}***')
        # churn_label = "Yes" if churn_probability > 0.5 else "No"
        # st.markdown(f'Churn prediction is ***{churn_label}***')

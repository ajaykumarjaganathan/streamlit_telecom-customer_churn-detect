import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'churn_model.pkl' is in the correct directory.")
        return None

# Create a simple model if none exists (for demo purposes)
def create_demo_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    data = {
        'tenure': [1, 72, 12, 24, 3, 60],
        'MonthlyCharges': [90.0, 30.0, 65.0, 50.0, 100.0, 45.0],
        'Contract': ['Month-to-month', 'Two year', 'One year', 'One year', 'Month-to-month', 'Two year'],
        'InternetService': ['Fiber optic', 'DSL', 'DSL', 'No', 'Fiber optic', 'DSL'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    
    # Simple encoding
    le = LabelEncoder()
    df['Contract'] = le.fit_transform(df['Contract'])
    df['InternetService'] = le.fit_transform(df['InternetService'])
    
    X = df.drop('Churn', axis=1)
    y = le.fit_transform(df['Churn'])
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save the demo model
    with open('churn_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model

# Main app function
def main():
    st.title("📱 Telecom Churn Predictor")
    st.write("Enter customer details to predict churn probability")
    
    # Load or create model
    model = load_model()
    if model is None:
        st.warning("Using a demo model with limited accuracy")
        model = create_demo_model()
    
    # User input form
    with st.form("customer_details"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 65.0)
            
        with col2:
            contract = st.selectbox("Contract Type", 
                                  ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", 
                                         ["DSL", "Fiber optic", "No"])
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare input data
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Simple encoding (match what the model expects)
        le = LabelEncoder()
        input_df['Contract'] = le.fit_transform(input_df['Contract'])
        input_df['InternetService'] = le.fit_transform(input_df['InternetService'])
        
        # Make prediction
        try:
            probability = model.predict_proba(input_df)[0][1]  # Probability of churn (Yes)
            
            # Display results
            st.subheader("Prediction Result")
            
            if probability > 0.5:
                st.error(f"High churn risk: {probability:.1%} probability")
                st.write("Suggested action: Offer retention incentives or improved service")
            else:
                st.success(f"Low churn risk: {probability:.1%} probability")
                st.write("Customer appears satisfied with current service")
            
            # Show probability gauge
            st.progress(probability)
            st.caption(f"Churn probability: {probability:.1%}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()

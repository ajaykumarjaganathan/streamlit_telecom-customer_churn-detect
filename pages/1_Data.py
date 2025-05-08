import pandas as pd
import streamlit as st
import os
from urllib.request import urlretrieve

def load_data():
    """Load the telecom churn dataset from GitHub with caching"""
    DATA_URL = "https://raw.githubusercontent.com/ajaykumarjaganathan/streamlit_telecom-customer_churn-detect/main/Churn%20Prediction%20Dataset.csv"
    LOCAL_PATH = "churn_data.csv"
    
    try:
        # Try to download the file if it doesn't exist locally
        if not os.path.exists(LOCAL_PATH):
            urlretrieve(DATA_URL, LOCAL_PATH)
            st.info("Downloaded dataset from GitHub")
        
        # Load the data with proper encoding
        df = pd.read_csv(LOCAL_PATH)
        
        # Basic data cleaning
        df.columns = df.columns.str.strip()  # Remove whitespace from column names
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info(f"Current directory: {os.getcwd()}")
        st.info(f"Files available: {os.listdir()}")
        return None

def main():
    st.title("Telecom Customer Churn Data Analysis")
    st.subheader("Dataset Overview")
    
    # Load data with progress indicator
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is not None:
        st.success("✅ Dataset loaded successfully!")
        
        # Show basic info
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show column information
        st.subheader("Column Information")
        st.json({
            "columns": list(df.columns),
            "dtypes": str(df.dtypes.to_dict())
        })
    else:
        st.warning("Failed to load dataset. Please check the connection.")

if __name__ == "__main__":
    main()

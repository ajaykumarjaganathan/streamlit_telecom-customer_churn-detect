import pandas as pd
import streamlit as st
import os

# URL of the dataset on GitHub
dataset_url = "https://github.com/ajaykumarjaganathan/streamlit_telecom-customer_churn-detect/blob/main/churn_dataset.csv?raw=true"

# Try to load the dataset from the GitHub URL
df = pd.read_csv(dataset_url)

# Proceed with your analysis as normal
st.title("Data Analysis")
st.header("Basic Information about the Dataset")
st.write(df.describe())

import pandas as pd
import streamlit as st
import os
from prettytable import PrettyTable

# Get the current working directory
current_dir = os.getcwd()

# Relative path to the dataset within the "dataset" folder
dataset_path = os.path.join(current_dir, 'dataset', 'Churn Prediction Dataset.csv')

# Load the dataset
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
info_table = PrettyTable()
info_table.field_names = ["Property", "Value"]
info_table.add_row(["Number of Rows", len(df)])
info_table.add_row(["Number of Columns", len(df.columns)])
info_table.add_row(["Column Names", ", ".join(df.columns)])
info_table.title = "Basic Information about the Dataset"

# Display summary statistics of numerical variables
summary_table = df.describe().T
summary_table = summary_table.reset_index()
summary_table.rename(columns={'index': 'Variable'}, inplace=True)
summary_table.title = "Summary Statistics of Numerical Variables"

# Display the first few rows of the dataset
head_table = df.head().to_string(index=False)

# Check for missing values
missing_values_table = PrettyTable()
missing_values_table.field_names = ["Column", "Missing Values"]
missing_values_table.align = "l"
for column in df.columns:
    missing_values_table.add_row([column, df[column].isnull().sum()])
missing_values_table.title = "Missing Values"

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
duplicate_table = PrettyTable()
duplicate_table.field_names = ["Duplicate Rows"]
duplicate_table.add_row([duplicate_count])
duplicate_table.title = "Duplicate Rows"

# Univariate Analysis
univariate_table = PrettyTable()
univariate_table.field_names = ["Variable", "Churn Distribution"]
univariate_table.add_row(["Churn", df['Churn'].value_counts()])
univariate_table.title = "Univariate Analysis - Churn Distribution"

# Bivariate Analysis
bivariate_table = PrettyTable()
bivariate_table.field_names = ["Variable", "Monthly Charges"]
bivariate_table.add_row(["Churn", df.groupby('Churn')['MonthlyCharges'].mean()])
bivariate_table.title = "Bivariate Analysis - Monthly Charges vs Churn"

# Display tables in Streamlit
st.title("Data Analysis")
st.header("Basic Information about the Dataset")
st.text(info_table.get_string())
st.header("Summary Statistics of Numerical Variables")
st.text(summary_table.to_string(index=False))
st.header("First Few Rows of the Dataset")
st.text(head_table)
st.header("Missing Values")
st.text(missing_values_table.get_string())
st.header("Duplicate Rows")
st.text(duplicate_table.get_string())
st.header("Univariate Analysis - Churn Distribution")
st.text(univariate_table.get_string())
st.header("Bivariate Analysis - Monthly Charges vs Churn")
st.text(bivariate_table.get_string())

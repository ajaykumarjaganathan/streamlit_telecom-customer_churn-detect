import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_metrics import metric, metric_row
import pygal
import leather
import plotly.express as px

# Load the dataset
dataset_path = 'dataset/Churn Prediction Dataset.csv'
df = pd.read_csv(dataset_path)

# Convert 'TotalCharges' column to numerical values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Set page title
st.set_page_config(page_title="Visualization Dashboard")

# Title for the page
st.title("Visualization Dashboard")

# Sidebar navigation
option = st.sidebar.selectbox(
    'Select:',
    ('Analytics Dashboard', 'Key Performance Indicators for Churn Prediction')
)

if option == 'Analytics Dashboard':
    # Research question 1: Distribution of churn for different Internet service types
    st.header("Research question 1: Distribution of churn for different Internet service types")

    # Using Plotly Express
    fig = px.bar(df, x='InternetService', color='Churn', barmode='group',
                title='Churn Distribution for Internet Service Types (Plotly Express)',
                category_orders={'InternetService': ['DSL', 'Fiber optic', 'No']},
                color_discrete_map={'No': 'lightgreen', 'Yes': 'yellow'})
    fig.update_xaxes(title="Internet Service Type")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig)

    # Research question 2: Impact of having a partner or dependents on customer churn
    st.header("Research question 2: Impact of having a partner or dependents on customer churn")

    # Using Altair
    partner_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Partner:O', title='Partner Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Partner Status (Altair)"
    )
    st.altair_chart(partner_chart, use_container_width=True)

    dependents_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Dependents:O', title='Dependents Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Dependents Status (Altair)"
    )
    st.altair_chart(dependents_chart, use_container_width=True)

    # Research question 3: Influence of contract type on customer churn
    st.header("Research question 3: Influence of contract type on customer churn")

    # Using Plotly Express
    fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group')
    fig2.update_layout(title="Churn Distribution for Contract Type (Plotly Express)")
    st.plotly_chart(fig2, use_container_width=True)

    # Research question 4: Impact of billing preference on customer churn
    st.header("Research question 4: Impact of billing preference on customer churn")

    # Convert 'Churn' column to boolean (0 for No, 1 for Yes)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Group data by Billing Preference and calculate churn
    billing_churn = df.groupby('PaperlessBilling')['Churn'].sum().reset_index()

    # Plot using Plotly Express
    fig = px.bar(billing_churn, x='PaperlessBilling', y='Churn', 
                labels={'PaperlessBilling': 'Billing Preference', 'Churn': 'Churn Count'},
                title='Churn Distribution for Billing Preference (Plotly Express)')
    st.plotly_chart(fig)



    # Using Altair
    gender_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('gender', title='Gender'),
        y=alt.Y('count()', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution by Gender (Altair)"
    )
    st.altair_chart(gender_chart, use_container_width=True)

    # Additional research questions
    st.header("Additional Research Questions")

    # Research question 6: Impact of tenure on customer churn
    st.header("Research question 6: Impact of tenure on customer churn")

    # Plot using Plotly Express
    fig = px.histogram(df, x='tenure', color='Churn', nbins=20,
                    labels={'tenure': 'Tenure', 'Churn': 'Churn'},
                    title='Impact of Tenure on Customer Churn')
    st.plotly_chart(fig)

    # Research question 7: Relationship between total charges and churn
    st.subheader("Research question 7: Relationship between total charges and churn")
    charges_churn_scatter = alt.Chart(df).mark_circle(size=60).encode(
        x='TotalCharges',
        y='Churn',
        color='Churn:N',
        tooltip=['TotalCharges', 'Churn']
    ).properties(
        title="Churn vs Total Charges (Altair)"
    ).interactive()
    st.altair_chart(charges_churn_scatter, use_container_width=True)


    
elif option == 'Key Performance Indicators for Churn Prediction':
    # Key Performance Indicators (KPIs)
    st.header("Key Performance Indicators (KPIs)")

    # Calculate Gross MRR Churn
    gross_mrr_churn = 0.05  # Example value, replace with actual calculation

    # Calculate Net MRR Churn
    net_mrr_churn = 0.03  # Example value, replace with actual calculation

    # Calculate Net Change in Customers
    net_change_customers = 100  # Example value, replace with actual calculation

    # Calculate Revenue Growth Rate
    revenue_growth_rate = 0.10  # Example value, replace with actual calculation

    # Calculate Activation Rate
    activation_rate = 0.75  # Example value, replace with actual calculation

    # Calculate DAU/MAU Ratio
    dau_mau_ratio = 0.65  # Example value, replace with actual calculation

    # Calculate Net Promoter Score (NPS)
    nps = 75  # Example value, replace with actual calculation

    # Calculate Customer Satisfaction Score (CSAT)
    csat = 85  # Example value, replace with actual calculation

    # Calculate Customer Lifetime Value (LTV)
    clv = 1500  # Example value, replace with actual calculation

    # Display metrics in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Metrics")
        metric("Gross MRR Churn", f"<span style='color:green'>{gross_mrr_churn}</span>")
        metric("Net MRR Churn", f"<span style='color:yellow'>{net_mrr_churn}</span>")
        metric("Net Change in Customers", f"<span style='color:blue'>{net_change_customers}</span>")
        metric("Revenue Growth Rate", f"<span style='color:pink'>{revenue_growth_rate}</span>")

    with col2:
        st.subheader("Product Metrics")
        metric("Activation Rate", f"<span style='color:purple'>{activation_rate}</span>")
        metric("DAU/MAU Ratio", f"<span style='color:orange'>{dau_mau_ratio}</span>")
        metric("Net Promoter Score (NPS)", f"<span style='color:green'>{nps}</span>")
        metric("Customer Satisfaction (CSAT)", f"<span style='color:yellow'>{csat}</span>")
        metric("Customer Lifetime Value (LTV)", f"<span style='color:blue'>{clv}</span>")
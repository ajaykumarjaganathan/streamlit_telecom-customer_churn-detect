import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('churn_dataset.csv')

# Data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# 1. Overall churn rate
total_customers = len(df)
churned_customers = df[df['Churn'] == 'Yes'].shape[0]
churn_rate = churned_customers / total_customers
print(f"Overall churn rate: {churn_rate:.1%}")

# 2. Demographic factors
# Senior citizens
senior_churn = df[df['SeniorCitizen'] == 1]['Churn'].value_counts(normalize=True)
print(f"\nSenior citizen churn rate: {senior_churn['Yes']:.1%}")

# Gender analysis
gender_churn = df.groupby('gender')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by gender:")
print(gender_churn)

# 3. Service factors
# Internet service
internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by internet service:")
print(internet_churn)

# Phone service
phone_churn = df.groupby('PhoneService')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by phone service:")
print(phone_churn)

# 4. Contract & billing
# Contract type
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by contract type:")
print(contract_churn)

# Payment method
payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by payment method:")
print(payment_churn)

# 5. Financial factors
# Monthly charges
monthly_charges = df.groupby('Churn')['MonthlyCharges'].mean()
print(f"\nAverage monthly charges:\nChurned: ${monthly_charges['Yes']:.2f}\nRetained: ${monthly_charges['No']:.2f}")

# Tenure
tenure = df.groupby('Churn')['tenure'].mean()
print(f"\nAverage tenure (months):\nChurned: {tenure['Yes']:.1f}\nRetained: {tenure['No']:.1f}")

# Visualization
plt.figure(figsize=(15, 10))

# Contract type churn
plt.subplot(2, 2, 1)
sns.barplot(x='Contract', y='Churn', data=df.replace({'Churn': {'Yes': 1, 'No': 0}}))
plt.title('Churn Rate by Contract Type')

# Internet service churn
plt.subplot(2, 2, 2)
sns.barplot(x='InternetService', y='Churn', data=df.replace({'Churn': {'Yes': 1, 'No': 0}}))
plt.title('Churn Rate by Internet Service')

# Payment method churn
plt.subplot(2, 2, 3)
sns.barplot(x='PaymentMethod', y='Churn', data=df.replace({'Churn': {'Yes': 1, 'No': 0}}))
plt.xticks(rotation=45)
plt.title('Churn Rate by Payment Method')

# Tenure distribution
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, element='step')
plt.title('Tenure Distribution by Churn Status')

plt.tight_layout()
plt.show()

# Additional useful visualizations
plt.figure(figsize=(15, 5))

# Monthly charges distribution
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn Status')

# Senior citizen churn
plt.subplot(1, 2, 2)
sns.barplot(x='SeniorCitizen', y='Churn', data=df.replace({'Churn': {'Yes': 1, 'No': 0}, 'SeniorCitizen': {1: 'Yes', 0: 'No'}}))
plt.title('Churn Rate by Senior Citizen Status')

plt.tight_layout()
plt.show()

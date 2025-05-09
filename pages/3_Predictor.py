import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Sample data creation matching your model's expectations
data = {
    'gender': ['Male', 'Female']*500,
    'SeniorCitizen': [0, 1]*500,
    'Partner': ['Yes', 'No']*500,
    'Dependents': ['Yes', 'No']*500,
    'tenure': list(range(1, 1001)),
    'PhoneService': ['Yes', 'No']*500,
    'MultipleLines': ['Yes', 'No', 'No phone service']*334,
    'InternetService': ['DSL', 'Fiber optic', 'No']*334,
    'OnlineSecurity': ['Yes', 'No', 'No internet service']*334,
    'OnlineBackup': ['Yes', 'No', 'No internet service']*334,
    'DeviceProtection': ['Yes', 'No', 'No internet service']*334,
    'TechSupport': ['Yes', 'No', 'No internet service']*334,
    'StreamingTV': ['Yes', 'No', 'No internet service']*334,
    'StreamingMovies': ['Yes', 'No', 'No internet service']*334,
    'Contract': ['Month-to-month', 'One year', 'Two year']*334,
    'PaperlessBilling': ['Yes', 'No']*500,
    'PaymentMethod': ['Electronic check', 'Mailed check', 
                     'Bank transfer (automatic)', 'Credit card (automatic)']*250,
    'MonthlyCharges': [20 + x%80 for x in range(1000)],
    'TotalCharges': [20 + x%80 * (x%72) for x in range(1000)],
    'Churn': ['Yes', 'No']*500
}

df = pd.DataFrame(data)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Create and save encoder
categorical_features = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

# Train and save model
model = RandomForestClassifier()
model.fit(X_processed, y)

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

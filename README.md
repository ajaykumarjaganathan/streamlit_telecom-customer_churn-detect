## IDG Customer Churn Predictor App

The IDG Customer Churn Predictor App is an interactive Streamlit application designed to predict customer churn based on provided data. This README provides comprehensive instructions on creating, deploying, and using the app.

### Table of Contents
1. [Setup](#setup)
2. [Running the App](#running-the-app)
3. [Usage](#usage)
    - [Home Page](#home-page)
    - [Login Page](#login-page)
    - [Data Page](#data-page)
    - [Predictor Page](#predictor-page)
    - [Dashboard Page](#dashboard-page)
    - [History Page](#history-page)
4. [Models Used](#models-used)
5. [Deployment](#deployment)
6. [Further Development](#further-development)
7. [Contributing](#contributing)
8. [License](#license)

### Setup <a name="setup"></a>

1. **Clone Repository**: Clone the repository containing the Streamlit app code.
2. **Install Dependencies**: Install the required dependencies using pip.
3. **Data Setup**: Ensure you have a CSV dataset named `Churn Prediction Dataset.csv` placed inside a folder named `dataset` in the project directory.
4. **Configuration**: Update the `config.yaml` file with necessary credentials and configuration details.

### Running the App <a name="running-the-app"></a>

To run the app locally, execute the following command in the project directory:

```bash
streamlit run app.py
```

The app will start running locally and can be accessed through a web browser.

### Usage <a name="usage"></a>

#### Home Page <a name="home-page"></a>
- Provides an overview of the app and its purpose.

#### Login Page <a name="login-page"></a>
- Existing Users: Enter your username and password to log in.
- New Users: Create an account by providing a new username and password.

#### Data Page <a name="data-page"></a>
- Displays basic information about the dataset.
- Shows summary statistics of numerical variables.
- Provides the first few rows of the dataset.
- Conducts univariate and bivariate analysis.
- Presents additional analysis using pandas styling.

#### Predictor Page <a name="predictor-page"></a>
- Batch Prediction: Upload a CSV dataset containing customer information to predict churn.
- Online Prediction: Input customer details interactively to predict churn.

#### Dashboard Page <a name="dashboard-page"></a>
- Provides visualizations and analytics related to customer churn.
- Includes research questions and key performance indicators.
- Offers insights through various charts and plots.

#### History Page <a name="history-page"></a>
- Tracks user interactions with the app.
- Displays a history log of actions performed by the user.
- Allows navigating back to previous points in history.

### Models Used <a name="models-used"></a>

#### Supported Models
1. Support Vector Machine (SVM)
2. XGBoost
3. Decision Tree Model
4. Random Forest



#### Description
- SVM: Supervised machine learning algorithm used for classification tasks.
- XGBoost: Implementation of gradient boosted decision trees designed for speed and performance.

#### Model Training
- Data Preprocessing
- Pipeline Creation
- Model Training
- Evaluation

#### Model Selection
- User Choice
- Performance Comparison

### Deployment 

- Model Serialization
- Model Loading

### Further Development <a name="further-development"></a>

- Model Tuning
- Model Expansion
- Model Monitoring

### Deployment 

-Performed app deployment with Render
https://idg-churn-prediction-app.onrender.com


### Contributing <a name="contributing"></a>

Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or create a pull request.

### License <a name="license"></a>

This project is licensed under the [Apache 2.0](LICENSE). Feel free to use, modify, and distribute the code for personal and commercial purposes.

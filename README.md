# ChurnSense – Customer Churn Prediction System

ChurnSense is an AI-powered customer churn prediction and analytics application built using Machine Learning and Streamlit. The system predicts whether a customer is likely to churn and supports data-driven customer retention decisions.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project applies machine learning techniques to analyse customer behaviour and identify customers who are at risk of leaving the service.

The project includes exploratory data analysis, data preprocessing, feature engineering, model training, model evaluation, and deployment through an interactive Streamlit web application.

## Machine Learning Workflow

- Performed Exploratory Data Analysis (EDA) to understand customer behaviour and churn patterns.
- Cleaned and preprocessed customer data.
- Performed feature engineering and categorical feature encoding.
- Trained and evaluated multiple machine learning models.
- Compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- Selected Random Forest as the final model based on evaluation performance.
- Achieved approximately 80.6% prediction accuracy.
- Built a production-ready machine learning pipeline.
- Integrated the trained model with a Streamlit web application.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Repository Structure

ChurnSense/
- app/
  - app.py
- churn_analysis.ipynb
- churn_model.pkl
- sample_customers.csv
- requirements.txt
- README.md
- .gitignore

## Features

- Customer churn prediction
- Interactive Streamlit web application
- Machine learning-based risk classification
- Customer data analysis
- Model comparison and evaluation
- Production-ready prediction pipeline

## Model Performance

- Integrated the trained Random Forest model with the Streamlit application for churn prediction.
Accuracy: 80.6%

The final model was selected after comparing multiple classification algorithms based on their evaluation performance.

## How to Run the Application

1. Clone the repository.
2. Install the required dependencies:

   pip install -r requirements.txt

3. Run the Streamlit application:

   streamlit run app/app.py

## Author

**Kirthi Patnaik**

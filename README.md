# Customer Churn Prediction

## Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company based on their personal information, account details, services, and billing information.

The main goal of this project is to help businesses identify customers who are at risk of churning so that appropriate customer retention strategies can be implemented.

## Problem Statement

Customer churn can cause significant losses for businesses. Instead of waiting for customers to leave, companies can use Machine Learning to identify customers who are more likely to churn.

This project builds a classification model that analyzes customer data and predicts whether the customer will:

- Stay with the company
- Leave the company

## Objectives

- Analyze customer data and identify churn patterns.
- Perform data cleaning and preprocessing.
- Perform Exploratory Data Analysis (EDA).
- Convert categorical data into numerical format.
- Train Machine Learning classification models.
- Evaluate model performance using different metrics.
- Predict whether a customer is likely to churn.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Dataset

The dataset contains customer-related information such as:

- Customer demographics
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn

### Target Variable

The target variable is `Churn`.

- `Yes` - Customer has left the company.
- `No` - Customer is still with the company.

## Machine Learning Workflow

The project follows the following workflow:

1. Data Collection
2. Data Loading
3. Data Cleaning
4. Handling Missing Values
5. Exploratory Data Analysis
6. Feature Engineering
7. Encoding Categorical Variables
8. Feature Scaling
9. Train-Test Split
10. Model Training
11. Model Evaluation
12. Churn Prediction

## Exploratory Data Analysis

Different visualizations are used to understand the relationship between customer characteristics and churn.

The analysis includes:

- Churn distribution
- Churn based on contract type
- Churn based on tenure
- Churn based on monthly charges
- Churn based on payment method
- Churn based on internet service
- Correlation analysis

## Machine Learning Models

The project can use classification algorithms such as:

### Logistic Regression

Logistic Regression is used for binary classification and provides a simple and interpretable baseline model.

### Decision Tree

Decision Tree classifies customers by creating a series of decision rules based on different features.

### Random Forest

Random Forest combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### XGBoost

XGBoost is a powerful gradient boosting algorithm that can provide strong performance on structured/tabular datasets.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Confusion Matrix

The confusion matrix contains:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

For customer churn prediction, Recall is an important metric because correctly identifying customers who are likely to churn can help businesses take preventive action.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   └── model.py
│
├── requirements.txt
├── README.md
└── .gitignore

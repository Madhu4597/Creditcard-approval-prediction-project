import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandasql as psql
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import math

warnings.filterwarnings('ignore')
st.title("Credit Card Approval Predictor")

# Load training data and train model on app startup
@st.cache_data
def load_and_train():
    # Load dataset
    df = pd.read_csv("Application_Data (credit card).csv")
    df_backup = df.copy()

    # Preprocessing
    df['Applicant_Gender'] = df['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)
    le = LabelEncoder()
    for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
        df[col] = le.fit_transform(df[col])

    # Features and target
    X = df.drop("Status", axis=1)
    y = df['Status']

    # Oversample minority class
    oversample = RandomOverSampler(sampling_strategy=0.125)
    X_over, y_over = oversample.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=42)

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model - using K=5 as example
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    return model, scaler, df_backup.columns.drop("Status")

model, scaler, feature_cols = load_and_train()

# File uploader for customer data
st.sidebar.header("Upload New Customer Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Customer Data")
    st.write(new_data.head())

    # Validate required columns
    missing_cols = set(feature_cols) - set(new_data.columns)
    if missing_cols:
        st.error(f"Missing columns in uploaded data: {missing_cols}")
    else:
        # Preprocessing: convert categorical variables similarly
        if 'Applicant_Gender' in new_data.columns:
            new_data['Applicant_Gender'] = new_data['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)
        for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
            if col in new_data.columns:
                new_data[col] = LabelEncoder().fit_transform(new_data[col])

        # Select and reorder features
        X_new = new_data[list(feature_cols)]

        # Scale features
        X_new_scaled = scaler.transform(X_new)

        # Predict approval status
        predictions = model.predict(X_new_scaled)
        new_data['Approval_Prediction'] = predictions
        new_data['Approval_Prediction_Label'] = new_data['Approval_Prediction'].map({0: 'Not Approved', 1: 'Approved'})

        st.subheader("Prediction Results")
        st.write(new_data[['Approval_Prediction_Label']])

        # Optionally download the results
        csv = new_data.to_csv(index=False).encode()
        st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv")

else:
    st.write("Upload customer data CSV on the left sidebar to get approval predictions.")


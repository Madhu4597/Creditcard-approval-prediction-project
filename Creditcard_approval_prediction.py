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

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

st.title("Credit Card Approval Prediction")

# Load dataset
credit_card = pd.read_csv("Application_Data(credit card).csv")
credit_card_bk = credit_card.copy()

# Display initial dataset sample
st.header("Initial Dataset Sample")
st.dataframe(credit_card.head())

# Data info and checks
st.header("Null Values in Dataset")
st.write(credit_card.isnull().sum())

st.header("Duplicate Entries in Dataset")
st.write(credit_card.duplicated().any())

st.header("Unique Values per Feature")
st.write(credit_card.nunique())

# Status distribution
st.header("Status Class Distribution")
Status_count = credit_card.Status.value_counts()
st.write(Status_count)
st.write(f"Class 0: {Status_count[0]}")
st.write(f"Class 1: {Status_count[1]}")
st.write(f"Proportion (1 to 0): {round(Status_count[1]/Status_count[0], 2)} : 1")
st.write(f"Total samples: {len(credit_card)}")

# Data preprocessing
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].str.replace('F', '0')
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].str.replace('M', '1')
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].astype(int)

# Encoding categorical variables
le = LabelEncoder()
for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
    credit_card[col] = le.fit_transform(credit_card[col])

st.header("Data Info After Encoding")
st.write(credit_card.info())

st.header("Descriptive Statistics")
st.write(credit_card.describe())

# Splitting features and target
IndepVar = [col for col in credit_card.columns if col != 'Status']
TargetVar = 'Status'
x = credit_card[IndepVar]
y = credit_card[TargetVar]

# Handle class imbalance with oversampling
oversample = RandomOverSampler(sampling_strategy=0.125)
x_over, y_over = oversample.fit_resample(x, y)
st.write(f"Oversampled data size: {x_over.shape}, {y_over.shape}")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.3, random_state=42)
st.write(f"Train/Test split shapes: {x_train.shape}, {x_test.shape}")

# Feature scaling
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

st.write(f"Class Distribution after Oversampling on train data:")
class_counts = pd.Series(y_train).value_counts()
st.write(class_counts)
st.write(f"Proportion (1 to 0): {round(class_counts[1]/class_counts[0], 2)} : 1")

# Train KNN model with default k=5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Dynamic input feature
st.sidebar.header("Enter Customer Data")

def user_input_features():
    gender = st.sidebar.selectbox('Applicant Gender', ['F', 'M'])
    income_type = st.sidebar.number_input('Income Type (encoded integer)', min_value=int(credit_card['Income_Type'].min()), max_value=int(credit_card['Income_Type'].max()), value=int(credit_card['Income_Type'].mean()))
    education_type = st.sidebar.number_input('Education Type (encoded integer)', min_value=int(credit_card['Education_Type'].min()), max_value=int(credit_card['Education_Type'].max()), value=int(credit_card['Education_Type'].mean()))
    family_status = st.sidebar.number_input('Family Status (encoded integer)', min_value=int(credit_card['Family_Status'].min()), max_value=int(credit_card['Family_Status'].max()), value=int(credit_card['Family_Status'].mean()))
    housing_type = st.sidebar.number_input('Housing Type (encoded integer)', min_value=int(credit_card['Housing_Type'].min()), max_value=int(credit_card['Housing_Type'].max()), value=int(credit_card['Housing_Type'].mean()))
    job_title = st.sidebar.number_input('Job Title (encoded integer)', min_value=int(credit_card['Job_Title'].min()), max_value=int(credit_card['Job_Title'].max()), value=int(credit_card['Job_Title'].mean()))
    
    # Please replace below numeric inputs with your actual numeric feature names and ranges
    feature1 = st.sidebar.number_input('Feature1', float(credit_card['Feature1'].min()), float(credit_card['Feature1'].max()), float(credit_card['Feature1'].mean()))
    feature2 = st.sidebar.number_input('Feature2', float(credit_card['Feature2'].min()), float(credit_card['Feature2'].max()), float(credit_card['Feature2'].mean()))
    
    data = {
        'Applicant_Gender': 0 if gender == 'F' else 1,
        'Income_Type': income_type,
        'Education_Type': education_type,
        'Family_Status': family_status,
        'Housing_Type': housing_type,
        'Job_Title': job_title,
        'Feature1': feature1,
        'Feature2': feature2,
        # Add other features here...
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale user input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
approval = 'Approved' if prediction[0] == 1 else 'Not Approved'

st.subheader('Prediction for Entered Customer')
st.write(f"The customer is predicted as: **{approval}**")

# Your existing KNN evaluation code goes here if needed

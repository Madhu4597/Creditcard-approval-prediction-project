import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math

warnings.filterwarnings("ignore")

st.title("Credit Card Approval Prediction")

# Load dataset
credit_card = pd.read_csv("Application_Data (credit card).csv")
credit_card_bk = credit_card.copy()

# Show initial data preview
st.header("Initial Dataset Sample")
st.dataframe(credit_card.head())

# Show data info
st.header("Data Info")
st.write(credit_card.info())

# Display the null values
st.header("Null Values in Dataset")
st.write(credit_card.isnull().sum())

# Duplicate values check
st.header("Duplicates in Dataset")
st.write(credit_card.duplicated().any())

# Class distribution
st.header("Status Class Distribution")
status_counts = credit_card['Status'].value_counts()
st.write(status_counts)
st.write(f"Proportion: {round(status_counts[1] / status_counts[0],2)} : 1")

# Preprocessing: Convert Gender to numeric
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)

# Encoding categorical columns
cols_to_encode = ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col])
    label_encoders[col] = le

# Features and target
X = credit_card.drop('Status', axis=1)
y = credit_card['Status']

# Oversampling to handle imbalance
ros = RandomOverSampler(sampling_strategy=0.125)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Sidebar: User inputs for prediction
st.sidebar.header('Input Customer Data')

def user_input_features():
    gender = st.sidebar.selectbox('Applicant Gender', ['F', 'M'])
    income_type = st.sidebar.selectbox('Income Type', label_encoders['Income_Type'].classes_)
    education_type = st.sidebar.selectbox('Education Type', label_encoders['Education_Type'].classes_)
    family_status = st.sidebar.selectbox('Family Status', label_encoders['Family_Status'].classes_)
    housing_type = st.sidebar.selectbox('Housing Type', label_encoders['Housing_Type'].classes_)
    job_title = st.sidebar.selectbox('Job Title', label_encoders['Job_Title'].classes_)
    
    # Numeric feature inputs (replace with your columns)
    feature1_min, feature1_max = X['Feature1'].min(), X['Feature1'].max()
    feature1 = st.sidebar.number_input('Feature1', feature1_min, feature1_max, float(X['Feature1'].mean()))
    
    feature2_min, feature2_max = X['Feature2'].min(), X['Feature2'].max()
    feature2 = st.sidebar.number_input('Feature2', feature2_min, feature2_max, float(X['Feature2'].mean()))
    
    data = {
        'Applicant_Gender': 0 if gender == 'F' else 1,
        'Income_Type': label_encoders['Income_Type'].transform([income_type])[0],
        'Education_Type': label_encoders['Education_Type'].transform([education_type])[0],
        'Family_Status': label_encoders['Family_Status'].transform([family_status])[0],
        'Housing_Type': label_encoders['Housing_Type'].transform([housing_type])[0],
        'Job_Title': label_encoders['Job_Title'].transform([job_title])[0],
        'Feature1': feature1,
        'Feature2': feature2,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Scale the input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
result = 'Approved' if prediction[0] == 1 else 'Not Approved'

st.subheader('Prediction')
st.write(f'The customer is {result}')

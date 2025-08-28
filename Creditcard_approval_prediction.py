import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
st.title("Credit Card Approval Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("Application_Data(credit card).csv")
    df['Applicant_Gender'] = df['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)
    le = LabelEncoder()
    for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
        df[col] = le.fit_transform(df[col])
    return df

data = load_data()
feature_cols = data.columns.drop('Status')
X = data[feature_cols]
y = data['Status']

# Balance data and train KNN model
oversample = RandomOverSampler(sampling_strategy=0.125)
X_over, y_over = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

st.sidebar.header("Choose Input Mode")

input_mode = st.sidebar.radio("Select input mode:", ['Single Customer Input', 'Upload CSV File'])

def preprocess_input(df):
    # Same preprocessing as training
    df['Applicant_Gender'] = df['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)
    le = LabelEncoder()
    for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
        df[col] = le.fit_transform(df[col])
    return df

if input_mode == 'Single Customer Input':
    st.sidebar.subheader("Enter Customer Details")
    gender = st.sidebar.selectbox("Applicant Gender", ['F', 'M'])
    income_type = st.sidebar.selectbox("Income Type", sorted(data['Income_Type'].unique()))
    education_type = st.sidebar.selectbox("Education Type", sorted(data['Education_Type'].unique()))
    family_status = st.sidebar.selectbox("Family Status", sorted(data['Family_Status'].unique()))
    housing_type = st.sidebar.selectbox("Housing Type", sorted(data['Housing_Type'].unique()))
    job_title = st.sidebar.selectbox("Job Title", sorted(data['Job_Title'].unique()))
    # Provide numeric inputs for each numeric feature - replace 'FeatureX' with actual names and ranges
    feature1 = st.sidebar.number_input("Feature1", float(data['Feature1'].min()), float(data['Feature1'].max()), float(data['Feature1'].mean()))
    feature2 = st.sidebar.number_input("Feature2", float(data['Feature2'].min()), float(data['Feature2'].max()), float(data['Feature2'].mean()))
    
    input_dict = {
        'Applicant_Gender': gender,
        'Income_Type': income_type,
        'Education_Type': education_type,
        'Family_Status': family_status,
        'Housing_Type': housing_type,
        'Job_Title': job_title,
        'Feature1': feature1,
        'Feature2': feature2
        # Add all other features here as necessary
    }
    input_df = pd.DataFrame([input_dict])
    input_df = preprocess_input(input_df)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    pred_label = 'Approved' if prediction[0] == 1 else 'Not Approved'
    st.subheader("Prediction Result")
    st.write(f"The customer is predicted as: **{pred_label}**")

else:
    st.sidebar.subheader("Upload Customer CSV file")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        new_customers = pd.read_csv(uploaded_file)
        st.write("Uploaded data", new_customers.head())
        
        # Validate columns
        missing = set(feature_cols) - set(new_customers.columns)
        if missing:
            st.error(f"Missing columns in uploaded file: {missing}")
        else:
            new_customers_processed = preprocess_input(new_customers)
            new_customers_scaled = scaler.transform(new_customers_processed)
            preds = model.predict(new_customers_scaled)
            new_customers['Approval_Prediction'] = preds
            new_customers['Status_Label'] = new_customers['Approval_Prediction'].map({0: 'Not Approved', 1: 'Approved'})
            st.subheader("Prediction Results")
            st.write(new_customers)
            csv = new_customers.to_csv(index=False).encode()
            st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv")


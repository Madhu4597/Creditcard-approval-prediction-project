import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("Credit Card Approval Prediction")

# Load data
credit_card = pd.read_csv("Application_Data(credit card).csv")

# Columns to use as features (excluding ID and Status)
ignore_cols = ["Applicant_ID", "Status"]
features = [col for col in credit_card.columns if col not in ignore_cols]

# Identify categorical and numeric features based on unique values and dtype
categorical_cols = ['Applicant_Gender', 'Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']
numeric_cols = [col for col in features if col not in categorical_cols]

# Encode categorical features and save encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col].astype(str))
    label_encoders[col] = le

X = credit_card[features]
y = credit_card['Status']

# Oversample and split
ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# --- Sidebar: dynamic user input ---
st.sidebar.header("Enter Customer Data")

def user_input_features():
    data = {}
    for col in categorical_cols:
        options = label_encoders[col].classes_
        value = st.sidebar.selectbox(f"{col.replace('_', ' ')}", options)
        data[col] = int(label_encoders[col].transform([value])[0])
    for col in numeric_cols:
        minval = float(credit_card[col].min())
        maxval = float(credit_card[col].max())
        meanval = float(credit_card[col].mean())
        value = st.sidebar.number_input(f"{col.replace('_', ' ')}", min_value=minval, max_value=maxval, value=meanval)
        data[col] = value
    return pd.DataFrame([data])[features]

input_df = user_input_features()

st.header('Entered Customer Data')
st.dataframe(input_df, use_container_width=True)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
result = "Approved" if prediction[0] == 1 else "Not Approved"

st.subheader('Prediction')
st.write(f"The customer is predicted as: **{result}**")

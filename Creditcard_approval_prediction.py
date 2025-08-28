import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

st.title("Credit Card Approval Prediction")

# Load dataset
credit_card = pd.read_csv("Application_Data(credit card).csv")

# Display all columns for user reference
st.header("Dataset Columns")
st.dataframe(pd.DataFrame({"Columns": credit_card.columns}), use_container_width=True)

# Preprocessing: Convert gender to numeric carefully, handle unknown/missing
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1})
credit_card['Applicant_Gender'] = pd.to_numeric(credit_card['Applicant_Gender'], errors='coerce').fillna(0).astype(int)

# Label encoding categorical variables and saving encoders for reuse
label_encoders = {}
categorical_cols = ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']
for col in categorical_cols:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col])
    label_encoders[col] = le

# Define features and target (exclude 'Status')
features = list(credit_card.columns)
features.remove('Status')
X = credit_card[features]
y = credit_card['Status']

# Handle class imbalance with oversampling
ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model 
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Sidebar user inputs for all features dynamically
st.sidebar.header("Enter Customer Data")

def user_input_features():
    # For categorical features show string labels for user-friendliness
    gender = st.sidebar.selectbox('Applicant Gender', ('F', 'M'))

    input_data = {'Applicant_Gender': 0 if gender == 'F' else 1}

    for col in categorical_cols:
        choices = list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(f"{col.replace('_',' ')}", choices)
        input_data[col] = label_encoders[col].transform([selected])[0]

    # Numeric features - find numeric columns by excluding categoricals, target
    numeric_cols = [col for col in features if col not in categorical_cols + ['Applicant_Gender']]

    for col in numeric_cols:
        min_val = float(credit_card[col].min())
        max_val = float(credit_card[col].max())
        mean_val = float(credit_card[col].mean())
        user_val = st.sidebar.number_input(col.replace('_', ' '), min_value=min_val, max_value=max_val, value=mean_val)
        input_data[col] = user_val

    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Show entered customer data
st.header("Entered Customer Data")
st.dataframe(input_df, use_container_width=True)

# Scale user input
input_scaled = scaler.transform(input_df)

# Predict and show result
prediction = model.predict(input_scaled)
result = 'Approved' if prediction[0] == 1 else 'Not Approved'

st.subheader("Prediction Result")
st.write(f"The customer is predicted to be: **{result}**")

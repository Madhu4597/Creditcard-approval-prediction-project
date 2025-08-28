import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
st.title("Credit Card Approval Prediction")

# --- Load dataset
credit_card = pd.read_csv("Application_Data(credit card).csv")
st.write("Your data columns are:", credit_card.columns)

# --- Preprocessing: Gender to numeric
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)

# --- Label encoding
label_encoders = {}
for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col])
    label_encoders[col] = le

# --- Features and target - replace below as needed
features_for_input = [
    'Applicant_Gender', 'Income_Type', 'Education_Type', 'Family_Status',
    'Housing_Type', 'Job_Title',
    # ADD your real numeric columns here:
    'Age', 'Annual_Income'   # <-- example REAL feature names
]

X = credit_card[features_for_input]
y = credit_card['Status']

# --- Oversample and split
ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# --- Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# --- Streamlit sidebar for manual input
st.sidebar.header('Enter Customer Data')

def user_input_features():
    gender = st.sidebar.selectbox('Applicant Gender', ['F', 'M'])
    income_type = st.sidebar.selectbox('Income Type', label_encoders['Income_Type'].classes_)
    education_type = st.sidebar.selectbox('Education Type', label_encoders['Education_Type'].classes_)
    family_status = st.sidebar.selectbox('Family Status', label_encoders['Family_Status'].classes_)
    housing_type = st.sidebar.selectbox('Housing Type', label_encoders['Housing_Type'].classes_)
    job_title = st.sidebar.selectbox('Job Title', label_encoders['Job_Title'].classes_)
    # --- For each real numeric column in features_for_input, add a number_input field:
    # (Replace 'Age', 'Annual_Income' with your real columns; adjust min/max/mean)
    age = st.sidebar.number_input("Customer Age", int(credit_card['Age'].min()), int(credit_card['Age'].max()), int(credit_card['Age'].mean()))
    income = st.sidebar.number_input("Annual Income", float(credit_card['Annual_Income'].min()), float(credit_card['Annual_Income'].max()), float(credit_card['Annual_Income'].mean()))
    
    data = {
        'Applicant_Gender': 0 if gender == 'F' else 1,
        'Income_Type': label_encoders['Income_Type'].transform([income_type]),
        'Education_Type': label_encoders['Education_Type'].transform([education_type]),
        'Family_Status': label_encoders['Family_Status'].transform([family_status]),
        'Housing_Type': label_encoders['Housing_Type'].transform([housing_type]),
        'Job_Title': label_encoders['Job_Title'].transform([job_title]),
        'Age': age,
        'Annual_Income': income
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Scale the input
input_scaled = scaler.transform(input_df)

# --- Predict
prediction = model.predict(input_scaled)
result = 'Approved' if prediction == 1 else 'Not Approved'

st.subheader('Prediction Result')
st.write(f'The customer is predicted as: **{result}**')


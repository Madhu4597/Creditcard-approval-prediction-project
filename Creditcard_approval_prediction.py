import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
st.title("Credit Card Approval Prediction")

# Load your CSV
credit_card = pd.read_csv("Application_Data(credit card).csv")
st.write("Columns for reference:", credit_card.columns.tolist())

# Preprocessing: Gender numeric
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1}).astype(int)

# Label encode categoricals and save encoder
label_encoders = {}
categorical_cols = ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']
for col in categorical_cols:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col])
    label_encoders[col] = le

# List all features except 'Status'
model_features = [
    'Applicant_Gender', 'Income_Type', 'Education_Type', 'Family_Status',
    'Housing_Type', 'Job_Title', 
    'Age', 'Salary', 'Credit_Amount'
] # <<<--- Replace with your real numeric columns

X = credit_card[model_features]
y = credit_card['Status']

# Oversample and train/test split
ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Sidebar - user input for ALL columns
st.sidebar.header('Enter New Customer Data')

def user_input_features():
    gender = st.sidebar.selectbox('Applicant Gender', ['F', 'M'])
    income_type = st.sidebar.selectbox('Income Type', list(label_encoders['Income_Type'].classes_))
    education_type = st.sidebar.selectbox('Education Type', list(label_encoders['Education_Type'].classes_))
    family_status = st.sidebar.selectbox('Family Status', list(label_encoders['Family_Status'].classes_))
    housing_type = st.sidebar.selectbox('Housing Type', list(label_encoders['Housing_Type'].classes_))
    job_title = st.sidebar.selectbox('Job Title', list(label_encoders['Job_Title'].classes_))
    
    age = st.sidebar.number_input('Age', int(credit_card['Age'].min()), int(credit_card['Age'].max()), int(credit_card['Age'].mean()))
    salary = st.sidebar.number_input('Salary', float(credit_card['Salary'].min()), float(credit_card['Salary'].max()), float(credit_card['Salary'].mean()))
    credit_amount = st.sidebar.number_input('Credit Amount', float(credit_card['Credit_Amount'].min()), float(credit_card['Credit_Amount'].max()), float(credit_card['Credit_Amount'].mean()))

    # Add any extra numeric features in same pattern

    data = {
        'Applicant_Gender': 0 if gender == 'F' else 1,
        'Income_Type': label_encoders['Income_Type'].transform([income_type])[0],
        'Education_Type': label_encoders['Education_Type'].transform([education_type])[0],
        'Family_Status': label_encoders['Family_Status'].transform([family_status])[0],
        'Housing_Type': label_encoders['Housing_Type'].transform([housing_type])[0],
        'Job_Title': label_encoders['Job_Title'].transform([job_title])[0],
        'Age': age,
        'Salary': salary,
        'Credit_Amount': credit_amount
        # Add more features if present
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
result = 'Approved' if prediction[0] == 1 else 'Not Approved'

st.subheader('Prediction Result')
st.write(f"The customer is predicted to be: **{result}**")

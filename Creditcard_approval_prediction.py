import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("Credit Card Approval Prediction")

# Load dataset directly from GitHub raw url
url = "https://raw.githubusercontent.com/Madhu4597/Creditcard-approval-prediction-project/main/Application_Data(credit card).csv"
credit_card = pd.read_csv(url)

# Drop ID and target column from features
ignore_cols = ['Applicant_ID', 'Status']
features = [col for col in credit_card.columns if col not in ignore_cols]

# Preprocess gender
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1})
credit_card['Applicant_Gender'] = pd.to_numeric(credit_card['Applicant_Gender'], errors='coerce').fillna(0).astype(int)

# Identify categorical and numeric columns
categorical_cols = ['Applicant_Gender', 'Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']
numeric_cols = [c for c in features if c not in categorical_cols]

# Label encode categorical columns and save encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col].astype(str))
    label_encoders[col] = le

# Separate features and target
X = credit_card[features]
y = credit_card['Status']

# Oversample minority class
ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Sidebar for dynamic user input
st.sidebar.header("Enter Customer Data")

def user_input_features():
    data = {}
    for col in features:
        if col in categorical_cols:
            options = list(label_encoders[col].classes_)
            selected = st.sidebar.selectbox(col.replace('_', ' '), options)
            data[col] = int(label_encoders[col].transform([selected])[0])
        else:
            min_val = float(credit_card[col].min())
            max_val = float(credit_card[col].max())
            mean_val = float(credit_card[col].mean())
            val = st.sidebar.number_input(
                col.replace('_', ' '),
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(mean_val),
                step=1,
                format="%d"
            )
            data[col] = val
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show entered data
st.header("Entered Customer Data")
st.dataframe(input_df)

# Scale inputs
input_scaled = scaler.transform(input_df)

# Predict approval
prediction = model.predict(input_scaled)
result = "Approved" if prediction[0] == 1 else "Not Approved"

st.subheader("Prediction Result")
st.write(f"The customer is predicted to be: **{result}**")

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("Credit Card Approval Prediction")

url = "https://raw.githubusercontent.com/Madhu4597/Creditcard-prediction-project/main/Application_Data%28credit%20card%29.csv"
credit_card = pd.read_csv(url)

st.write("Dataset columns:", credit_card.columns.tolist())

ignore_cols = ['Applicant_ID', 'Status']
features = [col for col in credit_card.columns if col not in ignore_cols]

# Preprocess Applicant Gender
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].replace({'F': 0, 'M': 1})
credit_card['Applicant_Gender'] = pd.to_numeric(credit_card['Applicant_Gender'], errors='coerce').fillna(0).astype(int)

categorical_cols = []
numeric_cols = []
for col in features:
    if credit_card[col].dtype == "object" or credit_card[col].nunique() < 20:
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    credit_card[col] = le.fit_transform(credit_card[col].astype(str))
    label_encoders[col] = le

X = credit_card[features]
y = credit_card['Status']

ros = RandomOverSampler(sampling_strategy=0.125)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

st.sidebar.header("Enter Customer Data")

def user_input_features():
    data = {}
    for col in features:
        if col == 'Applicant_Gender':
            gender = st.sidebar.selectbox('Applicant Gender', ['F', 'M'])
            data['Applicant_Gender'] = 0 if gender == 'F' else 1
        elif col in categorical_cols:
            options = label_encoders[col].classes_
            selected = st.sidebar.selectbox(col.replace('_', ' '), options)
            data[col] = int(label_encoders[col].transform([selected])[0])
        elif col in numeric_cols:
            val = st.sidebar.number_input(
                col.replace('_', ' '),
                min_value=int(credit_card[col].min()),
                max_value=int(credit_card[col].max()),
                value=int(credit_card[col].mean()),
                step=1,
                format="%d"
            )
            data[col] = val
    return pd.DataFrame([data])[features]

input_df = user_input_features()

st.write("Features used for prediction:", features)
st.write("User input dataframe:", input_df)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

result = 'Approved' if prediction[0] == 1 else 'Not Approved'
st.subheader("Prediction")
st.write(f"The customer is predicted as: **{result}**")


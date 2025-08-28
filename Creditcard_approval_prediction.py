# %% [markdown]
# Credit Card Approval Prediction (Streamlit App)

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

# Load Data
credit_card = pd.read_csv("Application_Data(credit card).csv")
credit_card_bk = credit_card.copy()

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

# Read previous results (optional: display)
KNN_Results = pd.read_csv("KNN_Results.csv")
EMResults1 = pd.read_csv("EMResults.csv")

st.header("Previous KNN Results Sample")
st.dataframe(KNN_Results.head())

# KNN Model loop with evaluation & ROC curve plot
st.header("KNN Algorithm Evaluation")

for k in range(1, 21):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(x_train, y_train)
    
    y_pred = model_knn.predict(x_test)
    y_pred_prob = model_knn.predict_proba(x_test)
    
    st.subheader(f"KNN with k={k}")
    actual = y_test
    predicted = y_pred
    matrix = confusion_matrix(actual, predicted, labels=[1,0])
    st.write("Confusion Matrix:\n", matrix)
    
    tp, fn, fp, tn = matrix.reshape(-1)
    
    sensitivity = round(tp/(tp+fn), 3)
    specificity = round(tn/(tn+fp), 3)
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
    balanced_accuracy = round((sensitivity+specificity)/2, 3)
    precision = round(tp/(tp+fp), 3) if (tp+fp)>0 else 0
    f1Score = round((2*tp/(2*tp + fp + fn)), 3)
    
    mx = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    MCC = round(((tp*tn) - (fp*fn)) / (math.sqrt(mx) if mx>0 else 1), 3)
    
    st.write(f"Accuracy: {accuracy*100:.2f} %")
    st.write(f"Precision: {precision*100:.2f} %")
    st.write(f"Recall (Sensitivity): {sensitivity*100:.2f} %")
    st.write(f"F1 Score: {f1Score}")
    st.write(f"Specificity: {specificity*100:.2f} %")
    st.write(f"Balanced Accuracy: {balanced_accuracy*100:.2f} %")
    st.write(f"MCC: {MCC}")
    st.write(f"ROC AUC Score: {roc_auc_score(actual, predicted):.3f}")
    
    fpr, tpr, _ = roc_curve(actual, y_pred_prob[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'KNN (K={k}) (area = {roc_auc_score(actual, predicted):.2f})')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for KNN')
    plt.legend(loc='lower right')
    st.pyplot(plt)
    plt.clf()
    
    new_row = {
        'Model Name': 'KNeighborsClassifier',
        'KNN K Value': k,
        'True_Positive': tp,
        'False_Negative': fn,
        'False_Positive': fp,
        'True_Negative': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': sensitivity,
        'F1 Score': f1Score,
        'Specificity': specificity,
        'MCC': MCC,
        'ROC_AUC_Score': roc_auc_score(actual, predicted),
        'Balanced Accuracy': balanced_accuracy
    }
    KNN_Results = pd.concat([KNN_Results, pd.DataFrame([new_row])], ignore_index=True)

st.subheader("Updated KNN Results")
st.dataframe(KNN_Results.head(20))

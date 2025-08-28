# %% [markdown]
# # Credit Card Approval Prediction (Cleaned Version)

# %%
#importing the libraries
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
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score)
import math

# ignore harmless warnings
warnings.filterwarnings("ignore")

# set to display all the columns in dataset
pd.set_option("display.max_columns", None)

# %%
# loading the dataset (adjust path according to where your file is in repo)
credit_card = pd.read_csv("data/Application_Data(credit card).csv")

# creating the backup file for the dataset
credit_card_bk = credit_card.copy()

# %%
# checking the first 5 records
print(credit_card.head())

# %%
# checking for null values
print(credit_card.isnull().sum())

# %%
# checking for duplicate values
print(credit_card.duplicated().any())

# %%
# Display the unique values of all the variables
print(credit_card.nunique())

# %%
# display the unique values by count for 'Status'
print(credit_card['Status'].value_counts())

# %%
# Count the target or dependent variable by '0' and '1' and their proportion
Status_count = credit_card.Status.value_counts()
print("Class 0: ", Status_count[0])
print("Class 1: ", Status_count[1])
print("Proportion: ", round(Status_count[1]/Status_count[0],2), ':1')
print("Total records: ", len(credit_card))

# %%
# info of the dataset
print(credit_card.info())

# %%
# replace 'Applicant_Gender' variable and convert to integer value
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].str.replace('F','0')
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].str.replace('M','1')
credit_card['Applicant_Gender'] = credit_card['Applicant_Gender'].astype(int)

# %%
# display unique counts for several categorical columns
print(credit_card['Income_Type'].value_counts())
print(credit_card['Education_Type'].value_counts())
print(credit_card['Family_Status'].value_counts())
print(credit_card['Housing_Type'].value_counts())
print(credit_card['Job_Title'].value_counts())

# %%
# Use LabelEncoder for several categorical variables
le = LabelEncoder()
for col in ['Income_Type', 'Education_Type', 'Family_Status', 'Housing_Type', 'Job_Title']:
    credit_card[col] = le.fit_transform(credit_card[col])

# %%
credit_card.info()

# %%
# Display descriptive statistics
print(credit_card.describe())

# %%
# Identify independent variables and target
IndepVar = [col for col in credit_card.columns if col != 'Status']
TargetVar = 'Status'

x = credit_card[IndepVar]
y = credit_card[TargetVar]

# %%
# Apply Random Over Sampling to balance dataset
oversample = RandomOverSampler(sampling_strategy=0.125)
x_over, y_over = oversample.fit_resample(x, y)

print(x_over.shape)
print(y_over.shape)

# %%
# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.30, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
# Scaling features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test)

# %%
# Check new class proportions after oversampling
Status_count = y_train.value_counts()
print("Proportion: ", round(Status_count[1]/Status_count[0],2), ':1')

# %%
# Read previous results files (adjust paths if needed)
KNN_Results = pd.read_csv("data/KNN_Results.csv")
EMResults1 = pd.read_csv("data/EMResults.csv")

# %%
# KNN algorithm implementation with evaluation
accuracy_list = []
for k in range(1, 21):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(x_train, y_train)
    
    y_pred = model_knn.predict(x_test)
    y_pred_prob = model_knn.predict_proba(x_test)
    
    print(f'KNN_K_value = {k}')
    print('Model Name: KNeighborsClassifier')
    
    actual = y_test
    predicted = y_pred
    matrix = confusion_matrix(actual, predicted, labels=[1,0])
    print('Confusion matrix : \n', matrix)
    
    tp, fn, fp, tn = matrix.reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)
    
    C_Report = classification_report(actual, predicted, labels=[1,0])
    print('Classification report : \n', C_Report)
    
    sensitivity = round(tp/(tp+fn), 3)
    specificity = round(tn/(tn+fp), 3)
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
    balanced_accuracy = round((sensitivity+specificity)/2, 3)
    precision = round(tp/(tp+fp), 3) if (tp+fp)>0 else 0
    f1Score = round((2*tp/(2*tp + fp + fn)), 3)
    
    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / (math.sqrt(mx) if mx>0 else 1), 3)
    
    print('Accuracy :', round(accuracy*100, 2), '%')
    print('Precision :', round(precision*100, 2), '%')
    print('Recall :', round(sensitivity*100, 2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100, 2), '%')
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2), '%')
    print('MCC :', MCC)
    
    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    fpr, tpr, _ = roc_curve(actual, y_pred_prob[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % roc_auc_score(actual, predicted))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
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

# %%
print(KNN_Results.head(20))

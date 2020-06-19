import streamlit as st
import numpy as np
import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import robust_scale
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn import random_projection
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import covariance
from sklearn.manifold import TSNE
from sklearn.neighbors.kde import KernelDensity as sclKDE
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import RandomUnderSampler
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
from sklearn.model_selection import KFold
from pylab import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from datetime import datetime
from IPython.display import display
import datetime as dt
import warnings
warnings.simplefilter('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


import requests
from requests.adapters import HTTPAdapter
import io
import os
import zipfile
import csv
import json
import pprint
import math


@st.cache
def get_transactions():
    # Given that the file is array of JSONs, initialize an empty list transactions.
    transactions = []
    
    with open('transactions.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        line_count = 0
        for row in csv_reader:
            transactions.append(json.loads(row[0])) 
            line_count += 1
        print("Processed {line_count} lines.".format(line_count=line_count))
        
    df = pd.DataFrame(transactions) 
    return df


@st.cache
def encode_variables(df):

    enc=LabelEncoder()

    for i in ToEncodeVars:
        df[[i]] = enc.fit_transform(df[[i]])
    
    enc_df=df.iloc[:,:]

    enc_df = enc_df.drop(columns=['currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange', 'accountNumber', 'customerId', 'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode', 'echoBuffer', 'currentBalance', 'merchantCity','merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'expirationDateKeyInMatch', 'availableMoney', 'enteredCVV'])

    return enc_df


@st.cache(allow_output_mutation=True)
def load_rf_grid_model():
    # load trained rf model
    rf_grid_model = pickle.load(open('insight/rf_grid.sav', 'rb'))
    return  rf_grid_model


@st.cache(allow_output_mutation=True)
def load_isvm_grid_model():
    # load trained svm model 
    svm_grid_model = pickle.load(open('insight/svm_grid.sav', 'rb'))
    return  svm_grid_model


def main():

    st.title('Abagnale: Cost sensitive fraud detection')
    st.text('Paro Guha')
    
    # Read the data
    df = get_transactions()
    df = df.copy(deep=True)

    df1 = df.drop(columns=[ 'accountNumber', 'customerId', 'transactionDateTime', 'acqCountry',
                            'merchantCountryCode', 'posEntryMode', 'posConditionCode',
                            'merchantCategoryCode', 'currentExpDate', 'accountOpenDate',
                            'dateOfLastAddressChange', 'enteredCVV', 'transactionType',
                            'echoBuffer', 'currentBalance', 'merchantCity', 'merchantState',
                            'merchantZip', 'cardPresent', 'posOnPremises', 'recurringAuthInd',
                            'expirationDateKeyInMatch', 'isFraud'])

    # ----------------------

    # Complete the call to convert the date column
    df['transactionDateTime'] =  pd.to_datetime(df['transactionDateTime'])
    df['currentExpDate'] =  pd.to_datetime(df['currentExpDate'])
    df['accountOpenDate'] =  pd.to_datetime(df['accountOpenDate'])
    df['dateOfLastAddressChange'] =  pd.to_datetime(df['dateOfLastAddressChange'])
    
    
    # Encode categorical variables
    
    from sklearn.preprocessing import LabelEncoder
    ToEncodeVars = ['accountNumber', 'customerId','merchantName',
           'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode',
           'merchantCategoryCode','transactionType', 'echoBuffer', 'merchantCity',
           'merchantState', 'merchantZip', 'cardPresent', 'posOnPremises',
           'recurringAuthInd', 'expirationDateKeyInMatch','cardCVV', 'enteredCVV', 'cardLast4Digits', 'isFraud']
    enc=LabelEncoder()
    
    
    # Transform categorical qualitative data into labels
    for i in ToEncodeVars:
        df[[i]] = enc.fit_transform(df[[i]])
    
    enc_data=df.iloc[:,:]
    
    # Dropping less important variables 
    data = enc_data.drop(columns=['currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange', 'accountNumber', 'customerId', 'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode', 'echoBuffer', 'currentBalance', 'merchantCity','merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'expirationDateKeyInMatch', 'availableMoney', 'enteredCVV'])
    
    
    # Feature engineering: Converting datetime to a new field seconds
    
    startTime = data.transactionDateTime.loc[0]
    endTime = data.transactionDateTime.loc[1]
    position = data.columns.get_loc('transactionDateTime')
    data['elapsed'] =  data.iloc[1:, position] - data.iat[0, position]
    seconds=data.elapsed.dt.total_seconds() 
    data['seconds'] = seconds
    data = data.drop(columns=['transactionDateTime', 'elapsed'])
    dataset = data.drop(columns=['isFraud']).fillna(0)
    target = data['isFraud'].fillna(0)

    # Splitting the Dataset and target into Break set and Testing set and stratifying for the minority class
    X_break, X_test, y_break, y_test = train_test_split(dataset, target, \
                                                        test_size=0.2, stratify=df['isFraud'],
                                                        random_state=42)
    
    # Splitting the Break set into training andvalidation set
    X_train, X_val, y_train, y_val = train_test_split(X_break, y_break, \
                                                        test_size=0.25, stratify=y_break,
                                                        random_state=42)
    
    # Standardize for the long tail of Dataset
    std_scale = StandardScaler().fit(X_train)
    
    # Scale data
    X_train_std = std_scale.transform(X_train)
    X_val_std = std_scale.transform(X_val)
    X_test_std = std_scale.transform(X_test)
    
    # Use Random undersampler from the majority class to correct for Imbalanced class model
    X_train_under, y_train_under = RandomUnderSampler(random_state=42).fit_sample(X_train_std,y_train)
    X_val_under, y_val_under = RandomUnderSampler(random_state=42).fit_sample(X_val_std,y_val)



    # -----------------------

    
    st.subheader('Dataset')
    st.write(df1.sample(5))

    st.subheader('Target')
    st.write(target.sample(5))

    # user input - interactice
    transactionAmount = \
         st.sidebar.slider('Transaction amount (USD): ',
                            min_value=0.0,
                            max_value=dataset['transactionAmount'].max(),
                            step=.01,
                            format='%f')
    creditLmit=st.sidebar.slider(
                            'Credit Limit (USD): ',
                            min_value=0,
                            max_value=int(dataset['creditLimit'].max()),
                            step=100,
                            format='%d')

    model_type = st.sidebar.selectbox("Recommendation Model", ["", "Random Forest"])
    if model_type == "Random Forest":
        st.subheader("Accuracy")
        rf_grid_model = load_rf_grid_model()
        result = rf_grid_model.score(X_test_std, y_test)
        st.write(result)
      
if __name__ == "__main__":
    main()

from django.shortcuts import render
import pandas as pd
import numpy as np


import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv(r'C:\\Train_Fraud_Detection.csv')
    df.replace('', np.nan, inplace=True)
    df = df.dropna()
    df[df['category'] == 'misc_net'] = 1
    df[df['category'] == 'grocery_pos'] = 2
    df[df['category'] == 'entertainment'] = 3
    df[df['category'] == 'gas_transport'] = 4
    df[df['category'] == 'misc_pos'] = 5
    df[df['category'] == 'grocery_net'] = 6
    df[df['category'] == 'shopping_net'] = 7
    df[df['category'] == 'shopping_pos'] = 8
    df[df['category'] == 'food_dining'] = 9
    df[df['category'] == 'personal_care'] = 10
    df[df['category'] == 'health_fitness'] = 11
    df[df['category'] == 'travel'] = 12
    df[df['category'] == 'kids_pets'] = 13
    df[df['category'] == 'home'] = 14
    df[df['gender'] == 'F'] = 1
    df[df['gender'] == 'M'] = 0
    X = df.drop(["trans_date_trans_time", "first", "last", "street", "city", "state", "zip", "job", "dob", "trans_num",
                 "unix_time", "merchant", "is_fraud"], axis='columns')
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lg_model = LogisticRegression()
    lg_model.fit(X_train, y_train)
    y_prediction = lg_model.predict(X_test)

    id = int(request.GET['n1'])
    cc_num = float(request.GET['n2'])
    cat = str(request.GET['n3'])
    amount = float(request.GET['n4'])
    gender = str(request.GET['n5'])
    lat = float(request.GET['n6'])
    long = float(request.GET['n7'])
    cpop = int(request.GET['n8'])
    merch_lat = float(request.GET['n9'])
    merch_long = float(request.GET['n10'])

    if cat == 'misc_net':
        c = 1
    if cat == 'grocery_pos':
        c = 2
    if cat == 'entertainment':
        c = 3
    if cat == 'gas_transport':
        c = 4
    if cat == 'misc_pos':
        c = 5
    if cat == 'grocery_net':
        c = 6
    if cat == 'shopping_net':
        c = 7
    if cat == 'shopping_pos':
        c = 8
    if cat == 'food_dining':
        c = 9
    if cat == 'personal_care':
        c = 10
    if cat == 'health_fitness':
        c = 11
    if cat == 'travel':
        c = 12
    if cat == 'kids_pets':
        c = 13
    if cat == 'home':
        c = 14
    else:
        c = 0

    if gender == 'Male' or gender == 'male':
        g = 1
    if gender == 'Female' or gender == 'female':
        g = 0
    else:
        c = 0

    i = (id, cc_num, c, amount, g, lat, long, cpop, merch_lat, merch_long)

    nparray = np.asarray(i)
    reshapedArray = nparray.reshape(1, -1)

    data = sc.transform(reshapedArray)
    y_prediction = lg_model.predict(data)

    result1 = ""

    if y_prediction[0] == 0:
        result1 = "Fraud Detected"
    else:
        result1 = "No Fraud Detected"

    return render(request, 'predict.html', {"result2": result1})

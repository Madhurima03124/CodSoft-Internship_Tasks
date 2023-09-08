from urllib import request

from django.shortcuts import render
import pandas as pd
import numpy as np

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
    df = pd.read_csv(r'C:\\Churn_dataset.csv')
    X = df.iloc[:, 3:-1]
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_prediction = rf_model.predict(X_test)

    cid = str(request.GET['n1'])
    surname = str(request.GET['n2'])
    crscore = int(request.GET['n3'])
    geo = str(request.GET['n4'])
    gender = str(request.GET['n5'])
    age = int(request.GET['n6'])
    tenure = int(request.GET['n7'])
    bal = float(request.GET['n8'])
    noofprod = float(request.GET['n9'])
    hascrcard = str(request.GET['n10'])
    isactive = str(request.GET['n11'])
    estsal = float(request.GET['n12'])


    if(hascrcard=='yes' or hascrcard=='Yes'):
        crcard=1
    else:
        crcard=0

    if (isactive == 'yes' or isactive== 'Yes'):
        ac = 1
    else:
        ac = 0

    if(geo=='Germany'):
        n9=1
    else:
        n9=0
    if(geo=='Spain'):
        n10=1
    else:
        n10=0
    if(gender=='Male'):
        n11=1
    else:
        n11=0

    i=(crscore,age,tenure,bal,noofprod,crcard,ac,estsal,n9,n10,n11)
    nparray = np.asarray(i)
    reshapedArray = nparray.reshape(1, -1)

    inputdata = sc.transform(reshapedArray)
    y_prediction = rf_model.predict(inputdata)
    result1=""

    if y_prediction[0] == 0:
       result1="Person doesn't exited"
    else:
       result1="Person Exited"
    return render(request,'predict.html',{"result2": result1})

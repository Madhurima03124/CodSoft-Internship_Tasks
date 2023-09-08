from pathlib import Path
import _ssl
import chardet
import nltk
import pandas as pd
from django.shortcuts import render
import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sympy.abc import y, Y

nltk.data.path.append("C:/nltk_data/nltk_data-gh-pages/packages/")
stopwords.words('english')
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    filename = r'C:\\sms_train_data.csv'
    detected = chardet.detect(Path(filename).read_bytes())
    # detected is something like {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

    encoding = detected.get("encoding")
    assert encoding, "Unable to detect encoding, is it a binary file?"

    sms_data = pd.read_csv(filename, encoding=encoding)
    sms_data.drop_duplicates(inplace=True)
    sms_data.reset_index(drop=True, inplace=True)
    clean = []
    ps = PorterStemmer()

    for i in range(0, sms_data.shape[0]):
        msg = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_data.v2[i])
        msg = msg.lower()
        words = msg.split()
        words = [word for word in words if word not in set(stopwords.words('english'))]
        words = [ps.stem(word) for word in words]
        msg = ' '.join(words)
        clean.append(msg)
    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(clean).toarray()
    Y = pd.get_dummies(sms_data['v1'])
    Y = Y.iloc[:, 1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, Y_train)
    y_prediction = classifier.predict(X_test)

    def predict_spam(sample_msg):
        sample_msg = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_msg)
        sample_msg = sample_msg.lower()
        sample_msg_words = sample_msg.split()
        sample_msg_words = [word for word in sample_msg_words if not word in set(stopwords.words('english'))]
        ps = PorterStemmer()
        final_msg = [ps.stem(word) for word in sample_msg_words]
        final_msg = ' '.join(final_msg)
        temp = cv.transform([final_msg]).toarray()
        return classifier.predict(temp)

    msgtext = str(request.GET['smstext'])
    result1 = ""

    if predict_spam(msgtext):
        result1 = "This is SPAM"
    else:
        result1 = "Normal Message"
    return render(request, 'predict.html', {"result2": result1})

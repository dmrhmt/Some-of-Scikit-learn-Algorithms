# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# veri yukleme


veriler = pd.read_csv("veriler.csv")
x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
#fit mi fit_transform mu ? fit:egitme, transform: uygulama. fit_transform: ogren ve uygula
#zaten ust satirda fit edilmiş, bir daha fit edip egitmezsin.
X_test = sc.transform(x_test)
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print("y_pred\n")
print(y_pred)
print("y_test\n")
print(y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)



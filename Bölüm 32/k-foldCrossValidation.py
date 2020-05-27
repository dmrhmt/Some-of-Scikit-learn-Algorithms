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


dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
#fit mi fit_transform mu ? fit:egitme, transform: uygulama. fit_transform: ogren ve uygula
#zaten ust satirda fit edilmiş, bir daha fit edip egitmezsin.
X_test = sc.transform(X_test)


from sklearn.svm import SVC

svc = SVC(kernel = "rbf", random_state=0)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("SVC Confusion Matrix rbf Kernel\n")
print(cm) 


from sklearn.model_selection import cross_val_score

#parametreler: 1.estimator: algoritma, 2. X, 3. Y, 4.cv: KATLAMA sayisi
cr_val_sc = cross_val_score(estimator=svc, X = X_train, y = y_train, cv = 4)

print(cr_val_sc.mean()) # ortalama basari
print(cr_val_sc.std()) # basaridaki standart sapma
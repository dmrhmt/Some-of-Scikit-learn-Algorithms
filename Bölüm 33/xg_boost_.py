# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:13:12 2020

@author: demir
"""
"""
XGBoost
--0---

- Yüksek verilerde iyi performans gösterir
- Hızlı çalışır(Hafızayı da iyi kullanıyor)
- Problem ve modelin yorumunun mümkün olması(Normalizasyon-Standartilazyon, verinin temizlenmesi, encodingler gibi olayları bypass edebiliyoruz)

"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
# veri yukleme


dataset = pd.read_csv("Churn_Modelling.csv")
#print(veriler)
# BURADA YINE ONEMLI BIR NOKTA VAR! ID, ISIM, TCKIMLIKNO GIBI UNIQUE SEYLER OVERFITTING'E YOK ACAR, UNUTMA! TRAIN'DE KOMPLE KULLANILMAZ, NE X'DE NE Y'DE!!



X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


#   LABEL ENCODER
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
print(X)

le = None
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)

# ONEHOT ENCODER
ohe = OneHotEncoder(categorical_features=[1]) # 1. kolonu al
X = ohe.fit_transform(X).toarray()
X = X[:,1:] # zaten 1. kolonu aldiydik, otesini al
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
print(cm)
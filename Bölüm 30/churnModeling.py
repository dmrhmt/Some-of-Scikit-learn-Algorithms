# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:20:49 2020

@author: demir
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
# veri yukleme


veriler = pd.read_csv("Churn_Modelling.csv")
#print(veriler)
# BURADA YINE ONEMLI BIR NOKTA VAR! ID, ISIM, TCKIMLIKNO GIBI UNIQUE SEYLER OVERFITTING'E YOK ACAR, UNUTMA! TRAIN'DE KOMPLE KULLANILMAZ, NE X'DE NE Y'DE!!



X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values


#   LABEL ENCODER
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
print(X)


le = None
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)





"""
# ONEHOT ENCODER
X_first = X[:,1]
X_rest = X[:,2:]

ohe = OneHotEncoder(handle_unknown="ignore") 
X_first = ohe.fit_transform(X_first).toarray()



X = pd.concat([X_first, X_rest], axis = 1)
"""


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# 3. YAPAY SINIR AGI

#import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu", input_dim=10)) # input_dim = giris katmani, 6 tane noronlu da bir gizli katman var
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu")) # zaten giris katmanını eklemistik, suan bir tane daha gizli katman ekliyoruz, yine 6 noronlu
classifier.add(Dense(1, kernel_initializer="uniform", activation="sigmoid")) # cikis katmani, 1 cikis noronu, sigmoid olsun activation func.
# asagidaki kod icin KESİNLİKLE, yukaridakiler icin ise opsiyonel fakat faydalı; DOKUMANTASYON OKU !!!!!
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # adam'dan baska optimizer'lar da var, zaten anlatti hoca !

classifier.fit(X_train, y_train, epochs=100)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > .5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

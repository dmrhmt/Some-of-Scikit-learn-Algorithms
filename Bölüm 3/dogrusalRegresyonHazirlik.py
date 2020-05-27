# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# veri yukleme


veriler = pd.read_csv("satislar.csv")
#print(veriler)

aylar = veriler[["Aylar"]]
satislar = veriler[["Satislar"]]

#print(aylar)
#print(satislar)

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
# model insaasi 
lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_test_tahmini = lr.predict(X_test)







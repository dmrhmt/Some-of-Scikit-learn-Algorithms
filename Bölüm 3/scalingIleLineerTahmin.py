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


# model insaasi 
lr = LinearRegression()
lr.fit(x_train, y_train)

y_test_tahmini = lr.predict(x_test)







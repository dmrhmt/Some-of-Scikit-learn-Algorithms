# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# veri yukleme


veriler = pd.read_csv("veriler.csv")


#ORTALAMAYA "NaN" STRINGINI DAHIL EDEMEYIZ, AYIRALIM
Yas = veriler.iloc[:,1:4].values
print(Yas) 

ulke = veriler.iloc[:,0:1].values
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

# ONEHOT ENCODER
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
#Artik LabelEncoder kullanmadan dogrudan OneHotEncoder kullanabilirsin

c = veriler.iloc[:,-1:].values
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print("cinsiyetin labelEncoder donusumu")
print(c)

ohe = OneHotEncoder(categorical_features="all")
c = ohe.fit_transform(c).toarray()
print("cinsiyetin ohe donusumu")
print(c)

dfUlke = pd.DataFrame(data = ulke, index = range(22), columns = ["fr", "tr", "us"]) 
#range(22): 0'dan 21'e kadar sayilarin bir listesini verir
#print(list(range(22)))
print(dfUlke)

dfYas = pd.DataFrame(data = Yas, index = range(22), columns = ["boy", "kilo", "yas"])
print(dfYas)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

dfCinsiyet = pd.DataFrame(data= c[:,0:1], index = range(22), columns = ["cinsiyet"])
print(dfCinsiyet)

sonuc = pd.concat([dfUlke, dfYas], axis = 1)
# axis = 1 diyoruz ki dogrudan alt alta yerlestirmek yerine kolon bazli yerlestirsin

sonuc2 = pd.concat([sonuc, dfCinsiyet], axis = 1)



x_train, x_test, y_train, y_test = train_test_split(sonuc, dfCinsiyet, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
 









# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# veri yukleme


veriler = pd.read_csv("eksikveriler.csv")
#print(veriler)


#   EKSIK VERILER

"""constructor icinde missing values'i "NaN" ile gosterecegimizi, stratejimizin
    ortalama almak oldugunu, axis ile satirda mi sutunda mi stratejinin
    uygulanacagini(sutun icin 0) belirtiyoruz.  
"""
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

#ORTALAMAYA "NaN" STRINGINI DAHIL EDEMEYIZ, AYIRALIM
Yas = veriler.iloc[:,1:4].values
print(Yas) 

ulke = veriler.iloc[:,0:1].values
"""print(ulke)"""

#   LABEL ENCODER
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

# ONEHOT ENCODER
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
#Artik LabelEncoder kullanmadan dogrudan OneHotEncoder kullanabilirsin

dfUlke = pd.DataFrame(data = ulke, index = range(22), columns = ["fr", "tr", "us"]) 
#range(22): 0'dan 21'e kadar sayilarin bir listesini verir
#print(list(range(22)))
print(dfUlke)

dfYas = pd.DataFrame(data = Yas, index = range(22), columns = ["boy", "kilo", "yas"])
print(dfYas)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

dfCinsiyet = pd.DataFrame(data= cinsiyet, index = range(22), columns = ["cinsiyet"])
print(dfCinsiyet)

sonuc = pd.concat([dfUlke, dfYas], axis = 1)
# axis = 1 diyoruz ki dogrudan alt alta yerlestirmek yerine kolon bazli yerlestirsin

sonuc2 = pd.concat([sonuc, dfCinsiyet], axis = 1)



x_train, x_test, y_train, y_test = train_test_split(sonuc, dfCinsiyet, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
 









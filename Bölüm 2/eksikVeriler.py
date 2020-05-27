# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer
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
#print(Yas) 
imputer = imputer.fit(Yas[:,1:])
print(Yas[:,1:4])
Yas[:,1:] = imputer.transform(Yas[:,1:])
print("yas Geliyor")
print(Yas)



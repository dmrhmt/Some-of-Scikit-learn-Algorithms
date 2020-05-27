#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


#veri on isleme

#encoder:  Kategorik -> Numeric
veriler2 = veriler.apply(LabelEncoder().fit_transform)
outlook = veriler2.iloc[:,:1].values
ohe = OneHotEncoder()
outlook_ohe=ohe.fit_transform(outlook).toarray()
#print("outlook")
#print(outlook)
outlook_df = pd.DataFrame(data = outlook_ohe, index = range(14), columns = ["overcast","rainy","sunny"])
sonveriler = pd.concat([outlook_df, veriler.iloc[:,1:3]], axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:], sonveriler], axis = 1)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:], test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



import statsmodels.api as sm
# 14 tane, 1 boyutlu, icinde 1 (int tipinde) olan array. Beta0 degerleri
X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis = 1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
# endog = y(bagimli degisken), exog = x'ler(bagimsiz degiskenler)
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)
r = r_ols.fit()
print(r.summary())
"""
#en yuksek p-value'yi eledim. X_l kismindaki rakamlar da 4'e kadar gitti.
sonveriler = sonveriler.iloc[:,1:]
import statsmodels.api as sm
# 14 tane, 1 boyutlu, icinde 1 (int tipinde) olan array. Beta0 degerleri
X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis = 1)
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
# endog = y(bagimli degisken), exog = x'ler(bagimsiz degiskenler)
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)
r = r_ols.fit()
print(r.summary())
"""

# AUTO BACKWARD ELIMINATION
"""
class Eliminations:
    def __init__(self, y):
        self.y = y
    
    
    def BackwardElimination(self, x, SL):    
        import statsmodels.formula.api as sm
        import numpy as np
        numVars = len(x.columns)
        numRows = len(x.index)
        temp = np.zeros((numRows,numVars)).astype(int)
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(self.y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        temp[:,j] = x[:, j]
                        x = np.delete(x, j, 1)
                        tmp_regressor = sm.OLS(self.y, x).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
                        if (adjR_before >= adjR_after):
                            x_rollback = np.hstack((x, temp[:,[0,j]]))
                            x_rollback = np.delete(x_rollback, j, 1)
                            print (regressor_OLS.summary())
                            return x_rollback
                        else:
                            continue
        regressor_OLS.summary()
        return x
    """

from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
X = pd.concat([x_train, x_test], axis = 0)
Y = pd.concat([y_train, y_test], axis = 0)




#TEKRAR PREDICT YAPTIRACAGIM
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
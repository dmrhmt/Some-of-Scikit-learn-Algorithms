# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:21:01 2020

@author: demir
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.svm import SVR
veriler = pd.read_csv("maaslar_yeni.csv")
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Linear Reg. OLS
print("Linear Reg. R2 Degeri X Icin\n")
print(r2_score(Y,lin_reg.predict(X)))
#OLS : ordinary least squares, lineer reg.da bilinmeyenleri tahmin etmek icin vs kullanılır
model_lin = sm.OLS(lin_reg.predict(X),X)
print(model_lin.fit().summary())


#Poly Reg.
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
#sutunlar sirasiyla x^0, x^1, x^2 olacak sekilde yazdiriyor
lin_reg2 = LinearRegression()
#x_poly ile y'yi ogren
lin_reg2.fit(x_poly, y)
#Poly Reg. OLS
print("POLY OLS")
model_poly = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model_poly.fit().summary())


#SVR Reg. Icin Scaling
sc = StandardScaler()
x_olcekli = sc.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

#SVR Reg.
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli)
#SVR Reg. OLS
print("SVR OLS")
model_svr = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model_svr.fit().summary())



#Decision Tree Reg.
r_dt = DecisionTreeRegressor(random_state=0)
#numpy array olan X ve Y'yi kullanarak ogrenmesini (fit etmesini) istiyoruz
r_dt.fit(X,Y)

print("Decision Tree OLS")
model_dt = sm.OLS(r_dt.predict(X),X)
print(model_dt.fit().summary())



#Random Forest Reg.
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y)
print("Random Forest OLS")
model_rf = sm.OLS(rf_reg.predict(X),X)
print(model_rf.fit().summary())

"""
Tek Parametreli
------0--------
Linear Reg.:
    R-squared: 0.942
Poly. Reg.:
    R-squared: 0.759
SVR:
    R-squared: 0.770
Decision Tree Reg.:
    R-squared: 0.751
Random Forest Reg.:
    R-squared: 0.719
++++++++++++++++++++++++++++++
3 Parametreli
------0--------
Linear Reg.:
    R-squared: 0.903
Poly. Reg.: 
    R-squared: 0.680
SVR:
    R-squared: 0.782
Decision Tree Reg.:
    R-squared: 0.679
Random Forest Reg.:
    R-squared: 0.713
++++++++++++++++++++++++++++++

"""
#pandas kut.den gelen corr() fonk. degiskenler arasi correlation matrix'i verir. 
print(veriler.corr())
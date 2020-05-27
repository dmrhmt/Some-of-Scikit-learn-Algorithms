# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:28:04 2020

@author: demir
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("maaslar.csv")
#dataframe dilimleme(slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
#numpy array donusumu
X = x.values
Y = y.values

#dogrusal(linear) model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#POLYNOMIAL REGRESSION

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
#sutunlar sirasiyla x^0, x^1, x^2 olacak sekilde yazdiriyor
lin_reg2 = LinearRegression()
#x_poly ile y'yi ogren
lin_reg2.fit(x_poly, y)


"""
POLY REGRESSION'da;

ilk olarak bir degree vererek regression oluşturuyoruz.
Bunu halihazırda değer olarak aldığımız x verisine göre 
fit_transform ediyoruz.
Ardından lineer regresyona önceden fit_transform edilmiş
x'i(x_poly) ve y yi verip fit ediyoruz.
scatter ile dikey ve yatay ekseni üzerine ne yazılacağını veriyoruz
plot ile ise noktaları çiziyoruz.
"""


poly_reg = PolynomialFeatures(degree = 19)
x_poly = poly_reg.fit_transform(X)
#sutunlar sirasiyla x^0, x^1, x^2 olacak sekilde yazdiriyor

lin_reg2 = LinearRegression()
#x_poly ile y'yi ogren
lin_reg2.fit(x_poly, y)

#gorsellestirme
plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

#tahminler


print(lin_reg.predict(X))
#print(lin_reg.predict(6.6))
print(lin_reg2.predict(poly_reg.fit_transform(X)))
print(lin_reg2.predict(poly_reg.fit_transform(X)))


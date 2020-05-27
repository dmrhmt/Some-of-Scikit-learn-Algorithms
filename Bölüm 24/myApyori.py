# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:14:09 2020

@author: demir
"""
import pandas as py
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("sepet.csv", header = None)

t = []
for i in range (0, 7501): # satir sayisi 7501, HARDCODED OLMAMALI! 
    
    t.append([str(veriler.values[i,j]) for j in range (0,20)]) # bir satirda max. 20 adet deger vardir dedik, HARDCODED OLMAMALI!

from apyori import apriori

kurallar = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)
print(list(kurallar))
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:09:06 2020

@author: demir
"""
#RANDOM SELECTION'DA ZEKA YOK ! SADECE RANDOM
import pandas as py
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")
"""
#RANDOM SELECTION
import random
toplam = 0
satirSayisi = len(veriler)
sutunSayisi = len(veriler.columns)
secilenler = []
for n in range(0, satirSayisi):
    randRow = random.randrange(sutunSayisi)
    secilenler.append(randRow)
    odul = veriler.values[n, randRow] # verilerdeki n. satirdaki daha onceden sectigimiz randRow'a bakiyor, 1 ise odul.
    toplam += odul
    
plt.hist(secilenler)
plt.show()
"""
satirSayisi = len(veriler) # 10 000 tiklama
sutunSayisi = len(veriler.columns)
birler = [0] * sutunSayisi
sifirlar = [0] * sutunSayisi
toplam = 0 # toplam odul
secilenler = []
import random
for n in range(0, satirSayisi):
    ad = 0 
    max_th = 0
    for i in range(0, sutunSayisi):
        randBeta =  random.betavariate(birler[i]+1, sifirlar[i]+1)
        if randBeta > max_th:
            max_th = randBeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad] # verilerdeki n. satirdaki daha onceden sectigimiz randRow'a bakiyor, 1 ise odul.
    if odul == 1:
        birler[ad] += 1
    else:
        sifirlar[ad] += 1    
    oduller[ad] += odul
    toplam += odul
print("toplam odul")
print(toplam)

plt.hist(secilenler)
plt.show()
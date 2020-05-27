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
#Ri(n)
oduller = [0] * sutunSayisi # 10 tane 0 iceren liste
#Ni(n)
tiklamalar = [0] * sutunSayisi
toplam = 0 # toplam odul
secilenler = []
import math
for n in range(0, satirSayisi):
    ad = 0 
    max_ucb = 0
    for i in range(0, sutunSayisi):
        if(tiklamalar[i] > 0): 
            ortalama_odul = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb = ortalama_odul + delta
        else:
            ucb = satirSayisi*10 # sadece ucb'yi cok buyuk bir sey yapmaya calisiyoruz
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] += 1
    odul = veriler.values[n, ad] # verilerdeki n. satirdaki daha onceden sectigimiz randRow'a bakiyor, 1 ise odul.
    oduller[ad] += odul
    toplam += odul
print("toplam odul")
print(toplam)

plt.hist(secilenler)
plt.show()
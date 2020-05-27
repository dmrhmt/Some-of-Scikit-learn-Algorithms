# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:42:43 2020

@author: demir
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = "k-means++")
kmeans.fit(X)
sonuclar = []
print(kmeans.cluster_centers_)

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++", random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11), sonuclar)
plt.show()


kmeans = KMeans(n_clusters=4, init = "k-means++", random_state=123)
Y_tahmin_kmeans = kmeans.fit_predict(X)

plt.scatter(X[Y_tahmin_kmeans == 0, 0], X[Y_tahmin_kmeans==0, 1], s = 100, c = "red")
plt.scatter(X[Y_tahmin_kmeans == 1, 0], X[Y_tahmin_kmeans==1, 1], s = 100, c = "blue")
plt.scatter(X[Y_tahmin_kmeans == 2, 0], X[Y_tahmin_kmeans==2, 1], s = 100, c = "green")
plt.scatter(X[Y_tahmin_kmeans == 3, 0], X[Y_tahmin_kmeans==3, 1], s = 100, c = "yellow")
plt.title("kmeans")
plt.show()


from sklearn.cluster import AgglomerativeClustering
agC = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
Y_tahmin = agC.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin==0, 1], s = 100, c = "red")
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin==1, 1], s = 100, c = "blue")
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin==2, 1], s = 100, c = "green")
plt.scatter(X[Y_tahmin_kmeans == 3, 0], X[Y_tahmin_kmeans==3, 1], s = 100, c = "yellow")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()
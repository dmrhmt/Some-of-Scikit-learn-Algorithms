# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:17:43 2020

@author: demir
"""

import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# veri yukleme


veriler = pd.read_csv("Wine.csv")
#print(veriler)
X = veriler.iloc[:,:13].values
Y = veriler.iloc[:,13].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test) # zaten fit ettik!

# PCA DONUSUMUNDEN ONCE GELEN LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# PCA DONUSUMUNDEN SONRA GELEN LOGICTIC REGRESSION
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

#tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("gercek vs PCA'siz")
print(cm)

cm2 = confusion_matrix(y_test, y_pred2)
print("gercek vs PCA'li")
print(cm2)

cm3 = confusion_matrix(y_pred, y_pred2)
print("pca'siz vs PCA'li")
print(cm3)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2)
# PCA'den farklı olarak y_train'i de veriyoruz ! Cunku LDA'in siniflari ogrenmesi gerekiyor(supervised), PCA'in ise boyle bir derdi yok
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LDA DONUSUMUNDEN SONRA GELEN LOGICTIC REGRESSION
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# LDA TAHMIN
y_pred_lda = classifier_lda.predict(X_test_lda)

cm4 = confusion_matrix(y_pred, y_pred_lda)
print("lda'siz vs LDA'li")
print(cm4)
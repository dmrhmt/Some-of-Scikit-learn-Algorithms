# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 08:24:16 2020

@author: demir
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv("Restaurant_Reviews.csv")

import re
import nltk
# eger metin ENG ise stop words bu kutuphaneden indirilebilir. Turkce icin ise burada var veya internette turkce stop words diye aratıp json, txt falan indirebilirsin
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download("stopwords")
derlem = []
for i in range(len(yorumlar)):
    #a-z arasi ve A-Z arasi filtreleme yapacakti, ^koyduk not oldu;yani artik a-z ve A-Z haricini filtreliyor. yanindaki bos string ise bunlari bosluk ile degistir demek
    yorum = re.sub("[^a-zA-Z]", " ",yorumlar["Review"][i])
    yorum = yorum.lower()
    #her kelimeyi listeye cevirelim
    yorum = yorum.split()
    
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derlem.append(yorum)
    
from sklearn.feature_extraction.text import CountVectorizer # feature extraction, bag of words
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()
y = yorumlar.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) # %72.5 acc
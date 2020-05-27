
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# veri yukleme


dataset = pd.read_csv("data_1.csv")
X = dataset.iloc[:,[0,2]].values
y = dataset.iloc[:,2].values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
#fit mi fit_transform mu ? fit:egitme, transform: uygulama. fit_transform: ogren ve uygula
#zaten ust satirda fit edilmiş, bir daha fit edip egitmezsin.
X_test = sc.transform(X_test)


from sklearn.svm import SVC

svc = SVC(kernel = "rbf", random_state=0)
fit_svc = svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svc)
print("SVC Confusion Matrix rbf Kernel\n")
print(cm) 


from sklearn.model_selection import cross_val_score

#parametreler: 1.estimator: algoritma, 2. X, 3. Y, 4.cv: KATLAMA sayisi
cr_val_sc = cross_val_score(estimator=fit_svc, X = X_train, y = y_train, cv = 4)

print(cr_val_sc.mean()) # ortalama basari
print(cr_val_sc.std()) # basaridaki standart sapma

# PARAMETRE OPTİMİZASYONU VE ALGORİTMA SECİMİ
from sklearn.model_selection import GridSearchCV
 # GIANT STEP- BABY STEP OLAYI. İlk büyük adımla başlarsın, optimize oldukça adım küçülmeli.
parameters = [{"C":[1,2,3,4,5], "kernel":["linear", "rbf"]},
               {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[1,0.5,0.1,0.01,0.001]}] # gamma default = auto (1/n)

#estimator = optimize edilecek algoritma
#param_grid = parametreler/denenecekler
#scoring = neye gore skorlanacak (ornegin accuracy)
#cv = kac katlamalı olacagı
#n_jobs = aynı anda çalışacak iş (paralel mak. ogrenmesi)
#paralel mak ogrenmesi icin; spark mllib, mahout gibi framework'lere bakılabilir 
"""from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier() # gini criterion da var
"""

from sklearn.tree import DecisionTreeClassifier
dec_classifier = DecisionTreeClassifier()
import numpy as np
dec_params = [{"min_samples_split":[np.arange(1.0,15.0,0.1).all()], "min_samples_leaf":[i for i in range(2,10)], "criterion":["gini", "entropy"],  "max_depth":[i for i in range (1,15)]},
               {"min_samples_split":[np.arange(15.1,30.0,0.1).all()], "min_samples_leaf":[i for i in range(10,20)], "criterion":["gini", "entropy"],  "max_depth":[i for i in range (15,30)]}]
 
gs_cv = GridSearchCV(estimator= dec_classifier, 
                     param_grid= dec_params,
                     scoring = "accuracy",
                     cv = 10)

gs_cv.fit(X_train, y_train)

best_result = gs_cv.best_score_
best_params = gs_cv.best_params_
best_index = gs_cv.best_index_
best_estimator = gs_cv.best_estimator_
error_score = gs_cv.error_score

best = best_estimator.fit(X_train, y_train)
best_train_pred = best.predict(X_train)
best_test_pred = best.predict(X_test)
from sklearn.metrics import f1_score

print("The training f1 score is:" , f1_score(y_train, best_train_pred))
print("The test f1 score is:" , f1_score(y_test, best_test_pred))

print("best result:")
print(best_result)
print("best params:")
print(best_params)
print("best index:")
print(best_index)
print("best estimator:")
print(best_estimator)
print("error score:")
print(error_score)

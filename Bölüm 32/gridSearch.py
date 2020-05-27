
"""
Created on Wed Jan 29 19:28:12 2020

@author: demir
"""
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# veri yukleme


dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
#fit mi fit_transform mu ? fit:egitme, transform: uygulama. fit_transform: ogren ve uygula
#zaten ust satirda fit edilmiş, bir daha fit edip egitmezsin.
X_test = sc.transform(X_test)


from sklearn.svm import SVC

svc = SVC(kernel = "rbf", random_state=0)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("SVC Confusion Matrix rbf Kernel\n")
print(cm) 


from sklearn.model_selection import cross_val_score

#parametreler: 1.estimator: algoritma, 2. X, 3. Y, 4.cv: KATLAMA sayisi
cr_val_sc = cross_val_score(estimator=svc, X = X_train, y = y_train, cv = 4)

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
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier() # gini criterion da var
rfc_params = [{"n_estimators":[1,10], "criterion":["gini", "entropy"]},
               {"n_estimators":[11, 501], "criterion":["gini", "entropy"]}]

gs_cv = GridSearchCV(estimator= rfc, 
                     param_grid= rfc_params,
                     scoring = "accuracy",
                     cv = 10)

gs_cv.fit(X_train, y_train)

best_result = gs_cv.best_score_
best_params = gs_cv.best_params_
best_index = gs_cv.best_index_
best_estimator = gs_cv.best_estimator_
error_score = gs_cv.error_score
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

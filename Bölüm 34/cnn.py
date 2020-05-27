# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:42:52 2020

@author: demir
"""

"""
class kerasImageClassifier:
    def __init__(self, training_set_directory, test_set_directory, image_target_size, )

"""

import keras
#  "samples_per_epoch=" + "weights.{size:04d}-" + 
checkpoint = keras.callbacks.ModelCheckpoint("epoch=" + "weights.{epoch:02d}-" + "loss=" + "{loss:.2f}" + "val_loss=" + "-{val_loss:.2f}.hdf5", monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [checkpoint]
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from keras.preprocessing.image import ImageDataGenerator

#normal resim okumaya yarayan kütüphaneden farkı ram'e tüm resimleri bir anda yüklemiyor, sırasıyla yüklüyor
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip= True)


training_set = train_datagen.flow_from_directory("veriler/training_set",
                                                     target_size=(64,64),
                                                     batch_size=1,
                                                     class_mode="binary")

test_set = test_datagen.flow_from_directory("veriler/test_set",
                                                     target_size=(64,64),
                                                     batch_size=1,
                                                     class_mode="binary")
import cv2
    
import os
shape = None
for root, dirs, files in os.walk("./veriler/training_set/erkek/", topdown=False):
    
    im = cv2.imread("./veriler/training_set/erkek/"+files[0])
    shape = im.shape
    print("shape" + str(shape))
    print("asdf")
    print(files[0])
    break
 
DIR =  "./veriler/test_set/"
sayi =  len([name for name in os.listdir(DIR+"erkek/") if os.path.isfile(os.path.join(DIR+"erkek/", name))])
sayi += len([name for name in os.listdir(DIR+"kadin/") if os.path.isfile(os.path.join(DIR+"kadin/", name))])



classifier = Sequential()

# Step 1 : Convolution
# 64x64 piksel, 3 renk katmanı var
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation="relu"))

# Step 2 : Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 : Second Loop of Convolution and Pooling
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 4 : Flattening
classifier.add(Flatten())

# Step 5 : N.N.
classifier.add(Dense(output_dim = 128, activation="relu"))
classifier.add(Dense(output_dim = 1, activation="sigmoid"))

# CNN
classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy"])


classifier.fit_generator(training_set,
                         samples_per_epoch = 1000,
                         nb_epoch = 3, # hiz kaygisi nedeniyle kucuk yapildi, su haliyle kotu, artirmalisin !
                         validation_data= test_set,
                         validation_steps = 2000,
                         callbacks=callbacks_list)

import numpy as np
import pandas as pd

test_set.reset()
pred = classifier.predict_generator(test_set, verbose= 1)
#pred = list(map(round,pred))


pred[pred > .5] = 1
pred[pred <= .5] = 0



print("prediction gecti")
#labels = (training_set.class_indicates)

test_labels = []

# 203 resim oldugu icin 203 yazilmis!
print("sayi"+str(sayi))
for i in range (0, sayi):
    test_labels.extend(np.array(test_set[i][1]))
    

print("test_labels")
print(test_labels)

#labels = (training_set.class_indicates)

"""
BURADA HOCANIN DEBUG KODLARI VAR, İŞİNE YARAYABİLİR!
İNTERNET SİTESİNDEN BULABİLİRSİN.

"""
file_names = test_set.filenames

sonuc = pd.DataFrame()
sonuc["dosyaisimleri"] = file_names
sonuc["tahminler"] = pred
sonuc["test"] = test_labels

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, pred)
print(cm)



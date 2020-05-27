# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
    
import os
shape = None
for root, dirs, files in os.walk("./veriler/training_set/erkek/", topdown=False):
    
    im = cv2.imread("./veriler/training_set/erkek/"+files[0])
    shape = im.shape()
    print("shape" + shape)
    print("asdf")
    print(files[0])
    break
 
DIR =  "./veriler/training_set/"
sayi =  len([name for name in os.listdir(DIR+"erkek/") if os.path.isfile(os.path.join(DIR+"erkek/", name))])
sayi += len([name for name in os.listdir(DIR+"kadin/") if os.path.isfile(os.path.join(DIR+"kadin/", name))])


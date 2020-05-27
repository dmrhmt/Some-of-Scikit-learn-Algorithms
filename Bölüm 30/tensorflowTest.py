# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:04:54 2020

@author: demir
"""

import tensorflow as tf
hello = tf.constant("Hello, Tensorflow")
sess = tf.Session()
print(sess.run(hello))
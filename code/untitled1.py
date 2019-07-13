#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:16:06 2019

@author: sebw
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

N_train = 60000
N_test = 10000
hid = 100
synapses = np.zeros((100, 784))
mnist_train = np.zeros((60000, 784))
mnist_test = np.zeros((10000,784))

hid_output_train = np.zeros((0, hid))
hid_output_test = np.zeros((0, hid))
a = np.dot(synapses, mnist_train[1])


print(a.shape)
print((a.reshape(1,100)).shape)
print(hid_output_train.shape)

for i in range(N_train):
    hid_output_train = np.concatenate((hid_output_train, (np.dot(synapses, mnist_train[i])).reshape(1, 100)), axis = 0)
for i in range(N_test):
    hid_output_test = np.concatenate((hid_output_test, (np.dot(synapses, mnist_test[i])).reshape(1, 100)), axis = 0)
    
print(hid_output_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:41:22 2019

@author: sebw
"""

import sys
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


n = float(sys.argv[1])
m = float(sys.argv[2])


hid_output_train_rand = np.load('hid_output_train_rand.npy')
hid_output_test = np.load('hid_output_test.npy')
label_train_rand = np.load('label_train_rand.npy')
label_test = np.load('label_test.npy')

hid = (np.shape(hid_output_train_rand))[1]
beta = float(sys.argv[3])

hid_output_train_rand = hid_output_train_rand ** n
hid_output_test = hid_output_test ** n

for i in range((np.shape(hid_output_train_rand))[0]):
    hid_output_train_rand[i] /= np.amax(hid_output_train_rand[i])
    
for i in range((np.shape(hid_output_test))[0]):
    hid_output_test[i] /= np.amax(hid_output_test[i])
    
hid_output_train_rand *= beta
hid_output_test *= beta
    
# Create output layer.
def customloss(y_True, y_Pred):
    diff = K.abs(y_True - y_Pred)
    return K.pow(diff, m)
opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)
opt2 = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)
opt3 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)
opt4 = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)
opt5 = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)

output = Sequential()
output.add(Dense(10, 
                 activation = 'tanh', 
                 input_dim = hid))
label_train_rand = keras.utils.to_categorical(label_train_rand, num_classes = 10)
label_test = keras.utils.to_categorical(label_test, num_classes = 10)

# Back-propagation training of output layer.
acc = np.zeros(0)
val_acc = np.zeros(0)

output.compile(optimizer = opt1, loss = customloss, metrics = ['accuracy'])
history = output.fit(hid_output_train_rand, label_train_rand, epochs = 100, batch_size = 100, validation_data = (hid_output_test, label_test))
acc = np.append(acc, (history.history)['acc'])
val_acc = np.append(val_acc, (history.history)['val_acc'])

output.compile(optimizer = opt2, loss = customloss, metrics = ['accuracy'])
history = output.fit(hid_output_train_rand, label_train_rand, epochs = 50, batch_size = 100, validation_data = (hid_output_test, label_test))
acc = np.append(acc, (history.history)['acc'])
val_acc = np.append(val_acc, (history.history)['val_acc'])

output.compile(optimizer = opt3, loss = customloss, metrics = ['accuracy'])
history = output.fit(hid_output_train_rand, label_train_rand, epochs = 50, batch_size = 100, validation_data = (hid_output_test, label_test))
acc = np.append(acc, (history.history)['acc'])
val_acc = np.append(val_acc, (history.history)['val_acc'])

output.compile(optimizer = opt4, loss = customloss, metrics = ['accuracy'])
history = output.fit(hid_output_train_rand, label_train_rand, epochs = 50, batch_size = 100, validation_data = (hid_output_test, label_test))
acc = np.append(acc, (history.history)['acc'])
val_acc = np.append(val_acc, (history.history)['val_acc'])

output.compile(optimizer = opt5, loss = customloss, metrics = ['accuracy'])
history = output.fit(hid_output_train_rand, label_train_rand, epochs = 50, batch_size = 100, validation_data = (hid_output_test, label_test))
acc = np.append(acc, (history.history)['acc'])
val_acc = np.append(val_acc, (history.history)['val_acc'])

#score = output.evaluate(hid_output_test, label_test, batch_size = 100)

error = np.ones(len(acc)) - acc
val_error = np.ones(len(val_acc)) - val_acc

error *= 100
val_error *= 100

fig2 = plt.figure(num = 2, figsize = (12.9, 10))
fig2, ax1 = plt.subplots()
ax1.plot(error, label = 'train')
ax1.plot(val_error, label = 'test')
ax1.legend()
plt.ylim(0, val_error[4])
plt.xlabel('Number of epochs')
plt.ylabel('Error %')
plt.title('n = {}'.format(n))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

fig2.savefig("output/fig4{}.pdf".format(str(sys.argv[4])))

#fig2.savefig("output/fig4{}.pdf".format('c'))

print("m = {}".format(m))

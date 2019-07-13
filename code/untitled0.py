#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:10:07 2019

@author: sebw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:08:27 2019

@author: sebw
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras

mnist = scipy.io.loadmat('mnist_all.mat')

Nc=10
N=784
N_train = 60000
N_test = 10000
n = 1


mnist_train=np.zeros((0,N))

for i in range(Nc):
    mnist_train=np.concatenate((mnist_train, mnist['train'+str(i)]), axis=0)
mnist_train=mnist_train/255.0 # 60000 * 784

mnist_test = np.zeros((0,N))
for i in range(Nc):
    mnist_test=np.concatenate((mnist_test, mnist['test'+str(i)]), axis=0)
mnist_test=mnist_test/255.0 # 10000 * 784


'''
Create labels.
'''
label_train = np.zeros(0)
label_test = np.zeros(0)
for i in range(Nc):
    label_train = np.append(label_train, np.full(((mnist['train'+str(i)]).shape)[0], i))
    label_test = np.append(label_test, np.full(((mnist['test'+str(i)]).shape)[0], i))


def draw_weights(synapses, Kx, Ky):
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()   
    

eps0=2e-2    # learning rate
Kx=10
Ky=10
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
mu=0.0
sigma=1.0
Nep=200      # number of epochs
Num=100      # size of the minibatch
prec=1e-30
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2
'''
Optimal given in paper: p=3, k=7, delta=0.4.
'''


fig=plt.figure(figsize=(12.9,10))


synapses = np.random.normal(mu, sigma, (hid, N))  # Each row a hidden unit, entries are weights
for nep in range(Nep):
    eps=eps0*(1-nep/Nep)
    mnist_train=mnist_train[np.random.permutation(N_train),:]
    for i in range(N_train//Num):
        inputs=np.transpose(mnist_train[i*Num:(i+1)*Num,:]) # Each column vector is the input vector to a cell
        sig=np.sign(synapses)
        tot_input=np.dot(sig*np.absolute(synapses)**(p-1),inputs) # I in equation 3
        # Each row contains the I values of one cell from each example in the mini-batch
        # Each column contains the I values of each cell from one example in the mini-batch
        
        y=np.argsort(tot_input,axis=0) # Sort each COLUMN, from small to large
        yl=np.zeros((hid,Num))
        yl[y[hid-1],np.arange(Num)]=1.0 # Set g(h) corresponding to largest I in each column to 1
        yl[y[hid-k],np.arange(Num)]=-delta
        # A row of yl contains the g(h) values for a cell given each example
        
        xx=np.sum(np.multiply(yl,tot_input),1) # Summing each row
        # xx is an 1-D array
        # xx is g(Q)*<W,v> in Eq.3
        ds=np.dot(yl,np.transpose(inputs)) - np.multiply(np.tile(xx.reshape((xx.shape[0],1)),(1,N)),synapses)
        # xx.shape[0]: number of entris in xx; xx.reshape: put entries of xx into a column; tile: repeat the column N times
        # second term is -g(Q)*<W,v>*W_i in Eq.3
        
        '''
        No idea.
        '''
        nc=np.amax(np.absolute(ds))
        if nc<prec:
            nc=prec
        synapses += eps*np.true_divide(ds,nc)
        
        
    draw_weights(synapses, Kx, Ky)


'''
Generate hidden layer output data with weights trained using unsupervised algorithm.
For 100-cell hidden unit: 60000 * 100 array for training, 10000 * 100 array for testing.
'''
hid_output_train = np.zeros((0, hid))
hid_output_test = np.zeros((0, hid))
for i in range(N_train):
    hid_output_train = np.concatenate((hid_output_train, np.power((np.dot(synapses, mnist_train[i])).reshape(1, hid), n)), axis = 0)
for i in range(N_test):
    hid_output_test = np.concatenate((hid_output_test, np.power((np.dot(synapses, mnist_test[i])).reshape(1, hid), n)), axis = 0)
    

'''
Output layer.
'''
output = Sequential()
output.add(Dense(10, activation = 'softmax'))
output.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrix = ['accuracy'])
label_train = keras.utils.to_categorical(label_train, num_classes = 10)
label_test = keras.utils.to_categorical(label_test, num_classes = 10)

output.fit(hid_output_train, label_train, epochs = 300, batch_size = 100)
score = output.evaluate(hid_output_test, label_test, batch_size = 100)

print(score)







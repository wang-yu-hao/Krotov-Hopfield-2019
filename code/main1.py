#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:08:27 2019

@author: sebw
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mnist = scipy.io.loadmat('mnist_all.mat')

# Number of classes.
Nc = 10
# Number of inputs.
N = 784
# Number of training and validation examples.
N_train = 60000
N_test = 10000
# n as in Eq. 1 and Fig. 4 of Krotov and Hopfield (2019).
#n = float(sys.argv[1])
# m as in Eq. 12 of Krotov and Hopfield (2019).
#m = float(sys.argv[2])
# Initial learning rate of unsupervised training.
eps0 = 4e-2
# Number of hidden units.
Kx = 40
Ky = 50
hid = Kx * Ky    
# Number of epochs for unsupervised training.
Nep = 200
# Size of the minibatch.
Num = 100
# Parameter that controls numerical precision of the weight updates.
prec = 1e-30
# Strength of the anti-Hebbian learning.
delta = 0.4
# Lebesgue norm of the weights.
p = 3.0
# Ranking parameter, must be integer that is bigger or equal than 2.
k = 7
# For radomly generating initial weights. Don't change.
mu = 0.0
sigma = 1.0
'''
Optimal hyperparameters given in paper for 2000-cell layer: eps0 = 4e-2, p=3.0, k=7, delta=0.4.
Working hyperparameters for 100-cell layer: eps0 = 4e-2, p=2.0, k=3, delta=0.4.
''' 
    

# Formatting training and validation examples.
mnist_train = np.zeros((0, N))
for i in range(Nc):
    mnist_train = np.concatenate((mnist_train, mnist['train'+str(i)]), axis = 0)
mnist_train = mnist_train / 255.0 # 60000 * 784

mnist_test = np.zeros((0, N))
for i in range(Nc):
    mnist_test = np.concatenate((mnist_test, mnist['test'+str(i)]), axis = 0)
mnist_test = mnist_test / 255.0 # 10000 * 784



# Create labels.
label_train = np.zeros(0)
label_test = np.zeros(0)
for i in range(Nc):
    label_train = np.append(label_train, np.full(((mnist['train'+str(i)]).shape)[0], i))
    label_test = np.append(label_test, np.full(((mnist['test'+str(i)]).shape)[0], i))

# Generate heat map of weights.
def draw_weights(synapses, Kx, Ky):
    yy = 0
    HM = np.zeros((28 * Ky, 28 * Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y * 28 : (y + 1) * 28, x * 28 : (x + 1) * 28] = synapses[yy, :].reshape(28, 28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM, cmap = 'bwr', vmin = -nc, vmax = nc)
    fig1.colorbar(im, ticks = [np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig1.canvas.draw()   
    
fig1=plt.figure(num = 1, figsize = (12.9, 10))

# Unsupervised training.
synapses = np.random.normal(mu, sigma, (hid, N))  # Each row a hidden unit, entries are weights
for nep in range(Nep):
    print("Epoch {}/{}".format(nep + 1, Nep))
    eps = eps0 * (1 - nep / Nep)
    mnist_train_rand = mnist_train[np.random.permutation(N_train), :]
    for i in range(N_train // Num):
        inputs = np.transpose(mnist_train_rand[i * Num : (i + 1) * Num, :]) # Each column vector is the input vector to a cell
        sig = np.sign(synapses)
        tot_input = np.dot(sig * np.absolute(synapses) ** (p - 1), inputs) # I in equation 3
        # Each row contains the I values of one cell from each example in the mini-batch
        # Each column contains the I values of each cell from one example in the mini-batch
        
        y = np.argsort(tot_input, axis = 0) # Sort each COLUMN, from small to large
        yl = np.zeros((hid, Num))
        yl[y[hid-1], np.arange(Num)] = 1.0 # Set g(h) corresponding to largest I in each column to 1
        yl[y[hid-k], np.arange(Num)] = -delta
        # A row of yl contains the g(h) values for a cell given each example
        
        xx = np.sum(np.multiply(yl, tot_input), 1) # Summing each row
        # xx is an 1-D array
        # xx is g(Q)*<W,v> in Eq.3
        ds = np.dot(yl, np.transpose(inputs)) - np.multiply(np.tile(xx.reshape((xx.shape[0], 1)), (1, N)), synapses)
        # xx.shape[0]: number of entris in xx; xx.reshape: put entries of xx into a column; tile: repeat the column N times
        # second term is -g(Q)*<W,v>*W_i in Eq.3
        
        nc=np.amax(np.absolute(ds))
        if nc < prec:
            nc = prec
        synapses += eps * np.true_divide(ds, nc)
            
    draw_weights(synapses, 6, 6)

print('Unsupervised training completed. Generating data for supervised training of output layer...')

# fig1.savefig("output/weights.pdf".format(hid))
fig1.savefig("output/weights.pdf")

# Generate hidden layer output data with weights trained using unsupervised algorithm.
# For 100-cell hidden unit: 60000 * 100 array for training, 10000 * 100 array for testing.
hid_output_test = np.zeros((0, hid))
hid_output_train_rand = np.zeros((0, hid))
label_train_rand = np.zeros(0)

rand1 = np.random.permutation(N_train)
for i in range(N_train):
    a_rand = (np.dot(synapses, mnist_train[rand1[i]])).reshape(1, hid)
    a_rand = np.maximum(a_rand, np.zeros(a_rand.shape))
    
    hid_output_train_rand = np.concatenate((hid_output_train_rand, a_rand), axis = 0)
    label_train_rand = np.append(label_train_rand, label_train[rand1[i]])
for i in range(N_test):
    a = (np.dot(synapses, mnist_test[i])).reshape(1, hid)
    a = np.maximum(a, np.zeros(a.shape))

    hid_output_test = np.concatenate((hid_output_test, a), axis = 0)
    

np.save('hid_output_train_rand', hid_output_train_rand)
np.save('hid_output_test', hid_output_test)
np.save('label_train_rand', label_train_rand)
np.save('label_test', label_test)


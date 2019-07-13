#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:10:07 2019

@author: sebw
"""


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('mnist_all.mat')
print(mat['test0'])
a = mat['test0']
d = a.shape
print(d)
print(sorted(mat.keys()))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:01:17 2019

@author: sebw
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,2,3,2,1])
b = np.array([5,6,7,8,9])

fig1 = plt.figure(0)
plt.plot(a)


fig2 = plt.figure(1)
plt.plot(b)

plt.figure(0)
plt.plot(b)

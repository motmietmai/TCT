# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:15:30 2018

@author: User
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)
w=np.dot(np.linalg.pinv(A),b)
x0=np.linspace(140,190,2)
y0=w[0]+w[1]*x0
plt.plot(X,y,'ro')
plt.plot(x0,y0)
plt.axis([140,190,45,75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

from sklearn import linear_model
regre=linear_model.LinearRegression(fit_intercept=False)
regre.fit(Xbar,y)
print(regre.coef_)


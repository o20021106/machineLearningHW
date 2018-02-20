# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:06:21 2017

@author: ipingou
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys
input_train=sys.argv[1]
#input_test=sys.argv[2]

train = np.loadtxt(input_train)
#train = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\train.txt")
#test = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\test.txt")
train_y = train[:,0].copy()
train_y[train_y>0] = -1
train_x = train[:,1:3].copy()
cost = np.array([10**-5,10**-3,10**-1,10**1,10**3])


def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]
ein = []
m = []
for i, c in enumerate(cost):
    m.insert(i, SVC(kernel = "linear",C = c))
    m[i].fit(train_x,train_y)
    ein.append(errorRate(train_x, train_y, m[i]))

w = []
for c in (m):
    w.append(np.dot(np.array(c.coef_),np.array(c.coef_.T))**.5)

w_length=np.array(w).reshape(5,)

plt.plot(np.array([-5,-3,-1,1,3]), w_length, 'ro')
plt.ylabel("||w||")
plt.xlabel("logC(base 10)")
plt.show()
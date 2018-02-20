# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:56:36 2017

@author: ipingou
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\train.txt")
test = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\test.txt")
train_y = train[:,0].copy()
train_y[train_y!=8] = -1
train_y[train_y==8] = 1
train_x = train[:,1:3].copy()

test_y = test[:,0].copy()
test_y[test_y!=8] = -1
test_y[test_y==8] = 1
test_x = test[:,1:3].copy()

def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]

cost = np.array([10**-5,10**-3,10**-1,10**1,10**3])

m = []
ein=[]

for i, c in enumerate(cost):
    model = SVC(kernel = "poly",degree = 2, coef0 = 1, gamma = 1,C = c)
    model.fit(train_x,train_y)
    m.append(model)
    ein.insert(i,errorRate(train_x, train_y, model)) 

plt.plot(np.array([-5,-3,-1,1,3]), ein, 'ro')
plt.ylabel("Ein")
plt.xlabel("logC(base 10)")
plt.show()


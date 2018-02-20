# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:59:45 2017

@author: ipingou
"""


import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys
input_train=sys.argv[1]

train = np.loadtxt(input_train)
train_y = train[:,0].copy()
train_y[train_y!=0] = -1
train_y[train_y==0] = 1
train_x = train[:,1:3].copy()


def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]

cost = np.array([10**-3,10**-2,10**-1,10**0,10**1])

m = []
n_sv=[]
Alpha = []
sv = []
free_sv=[]

for i, c in enumerate(cost):
    model = SVC(kernel = "rbf", gamma = 80,C = c)
    model.fit(train_x,train_y)
    m.append(model)
    n_sv.append(sum(model.n_support_))
    #alpha is the lagrange multiplier
    alpha = np.absolute(model.dual_coef_)
    Alpha.append(alpha)
    support = model.support_.copy()
    #support is the indices of support vectors
    support = support.reshape(1, model.support_.shape[0])
    sv.append(support)
    #free_sv is the indices of free support vectors
    free_sv.append(support[alpha<c])

def distance_c(w):
    return 1/np.dot(w,w.T)

w = []
Distance = []

for i,j in zip(Alpha, sv):
    y = train_y[j]
    x = train_x[j]
    x = x.reshape((x.shape[1],x.shape[2]))
    x_transpose = x.T
    weight = np.dot(x_transpose,np.multiply(i,y).T)
    weight = weight.flatten()
    w.append(weight)
    Distance.append(distance_c(weight))

plt.plot(np.array([-5,-3,-1,1,3]), Distance, 'ro')
plt.ylabel("Distance")
plt.xlabel("logC(base 10)")
plt.show()

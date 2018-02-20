# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:57:13 2017

@author: ipingou
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys
input_train=sys.argv[1]
input_test=sys.argv[2]


train = np.loadtxt(input_train)
test = np.loadtxt(input_test)

train_y = train[:,0].copy()
train_y[train_y!=0] = -1
train_y[train_y==0] = 1
train_x = train[:,1:3].copy()

test_y = test[:,0].copy()
test_y[test_y!=8] = -1
test_y[test_y==8] = 1
test_x = test[:,1:3].copy()

def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]

gamma = np.array([10**0,10**1,10**2,10**3,10**4])

m = []
n_sv=[]
Alpha = []
sv = []
free_sv=[]
eout = []

for i, g in enumerate(gamma):
    model = SVC(kernel = "rbf", gamma = g,C = 0.1)
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
    free_sv.append(support[alpha<0.1])
    eout.append(errorRate(test_x,test_y,model))


plt.plot(np.array([0,1,2,3,4]), eout, 'ro')
plt.ylabel("Eout")
plt.xlabel("logGamma(base 10)")
plt.show()

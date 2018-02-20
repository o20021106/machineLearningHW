# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:42:49 2017

@author: ipingou
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys

input_train=sys.argv[1]

train = np.loadtxt(input_train)
train_y = train[:,0].copy()
train_y[train_y!=8] = -1
train_y[train_y==8] = 1
train_x = train[:,1:3].copy()


def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]

cost = np.array([10**-5,10**-3,10**-1,10**1,10**3])

m = []
n_sv=[]

for i, c in enumerate(cost):
    model = SVC(kernel = "poly",degree = 2, coef0 = 1, gamma = 1,C = c)
    model.fit(train_x,train_y)
    m.append(model)
    n_sv.append(sum(model.n_support_))

plt.plot(np.array([-5,-3,-1,1,3]), n_sv, 'ro')
plt.ylabel("number of support vectors")
plt.xlabel("logC(base 10)")
plt.show()

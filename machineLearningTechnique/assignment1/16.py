# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:26:21 2017

@author: ipingou
"""
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

train = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\train.txt")
test = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\test.txt")
train_y = train[:,0].copy()
train_y[train_y!=0] = -1
train_y[train_y==0] = 1
train_x = train[:,1:3].copy()

def errorRate(X,Y,m):
    return  sum(m.predict(X)!=Y)/X.shape[0]

gamma = np.array([10**-1,10**0,10**1,10**2,10**3])
Gamma = []

for i in range(100):
    index = random.sample(range(0,train.shape[0]-1),1000)
    train_xx = train_x.copy()
    val_xx =train_xx[index]
    train_xx = np.delete(train_xx,index,0)
   
    train_yy = train_y.copy()
    val_yy =train_yy[index]
    train_yy = np.delete(train_yy,index)
    
    Eval = np.array([])
    m = []
    for i, g in enumerate(gamma):
        model = SVC(kernel = "rbf", gamma = g,C = 0.1)
        model.fit(train_xx,train_yy)
        m.append(model)
        Eval = np.append(Eval, errorRate(val_xx,val_yy,model))
   #print(Eval)
    Gamma.append(np.argmin(Eval))

Gamma2 = np.array(Gamma)
Gamma2[Gamma2 == 0] = -1
Gamma2[Gamma2 == 1] = 0
Gamma2[Gamma2 == 2] = 1
Gamma2[Gamma2 == 3] = 2
Gamma2[Gamma2 == 4] = 3
u, count = np.unique(np.array(Gamma2),return_counts = True)

plt.bar(u,count,0.5)
plt.title("Histogram")
plt.xlabel("log Gamma")
plt.ylabel("Frequency")
plt.xlim([-2,4])
plt.show()

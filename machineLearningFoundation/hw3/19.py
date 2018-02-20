# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:18:35 2017

@author: ipingou
"""

import numpy as np
import math
import matplotlib.pyplot as plt

k="C:/Users/ipingou/OneDrive/文件/碩二上/ml/hw3/hw4_train.dat"
l="C:/Users/ipingou/OneDrive/文件/碩二上/ml/hw3/hw4_test.dat"
data = np.loadtxt(k)
features = data.shape[1]
size = data.shape[0]
X = data[:,0:features-1]
X = np.insert(X,0,1,axis=1)
Y = data[:,-1]
Y=Y.reshape(len(Y),1)

X_cv = np.split(X,5,0)
Y_cv = np.split(Y,5,0) 

data_test = np.loadtxt(l)
features_test = data_test.shape[1]
size_test = data_test.shape[0]
X_test = data_test[:,0:features_test-1]
X_test = np.insert(X_test,0,1,axis=1)
Y_test = data_test[:,-1]
Y_test = Y_test.reshape(len(Y_test),1)

def error(w,X,Y):
    predict = np.sign(np.dot(X,w))
    return (1/X.shape[0])*sum(predict!=Y)
    
def train(L,X,Y):
    Xt = np.transpose(X)
    I=np.identity(X.shape[1])
    XtY=np.dot(Xt,Y)
    return np.dot(np.linalg.inv(np.dot(Xt,X)+L*I),XtY) 

#16
def cv_error(L,X,Y):
    total = 0    
    for i in range(5):
        X_val = X_cv[i]
        Y_val = Y_cv[i]
        index=np.array(range(5))!=i
        X_train = [i for (i, v) in zip(X_cv, index) if v]
        Y_train = [i for (i, v) in zip(Y_cv, index) if v]
        X_train = np.concatenate(X_train,axis=0)
        Y_train = np.concatenate(Y_train,axis=0)        
        w = train(L,X_train,Y_train)
        total = total + error(w,X_val, Y_val)
    return total/5

L_list=np.power(10,np.array(range(-10,3)),dtype=float)

Ecv_list =[]

for L in L_list:
    Ecv_list.append(cv_error(L,X_cv,Y_cv))           

plt.plot(np.array(range(-10,3)),Ecv_list,'ro')
plt.show

    
minEcv = min(Ecv_list)
Lmin_cv=max((L_list.reshape(len(L_list),1))[Ecv_list==minEcv])
print("Lambda with minumn Ecv: ", Lmin_cv,"\nEcv: ",minEcv)

#20

w = train(Lmin_cv, X,Y)

Ein = error(w, X,Y)
Eout = error(w,X_test, Y_test)
print("Ein: {} \n Eout: {}".format(Ein, Eout))
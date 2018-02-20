# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:18:35 2017

@author: ipingou
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
input_train=sys.argv[1]
input_test=sys.argv[2]


def preprocess(file):
    data = np.loadtxt(file)
    features = data.shape[1]
    X = data[:,0:features-1]
    X = np.insert(X,0,1,axis=1)
    Y = data[:,-1]
    Y=Y.reshape(len(Y),1)
    return [X,Y]

def error(w,X,Y):
    predict = np.sign(np.dot(X,w))
    return (1/X.shape[0])*sum(predict!=Y)   

def train(L,X,Y):
    Xt = np.transpose(X)
    I=np.identity(X.shape[1])
    XtY=np.dot(Xt,Y)
    return np.dot(np.linalg.inv(np.dot(Xt,X)+L*I),XtY) 
    
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
    
[X,Y]=preprocess(input_train)
[X_test, Y_test] = preprocess(input_test)
[X_train,X_val] = np.split(X, np.array([120]))
[Y_train, Y_val] = np.split(Y, np.array([120]))
X_cv = np.split(X,5,0)
Y_cv = np.split(Y,5,0) 

L_list_log = list(range(-10,3))
L_list=np.power(10,L_list_log,dtype=float)

Ecv_list =[]
for L in L_list:
    Ecv_list.append(cv_error(L,X_cv,Y_cv))           

plt.plot(L_list_log,Ecv_list,'ro')
plt.show
    
minEcv = min(Ecv_list)
Lmin_cv=max((L_list.reshape(len(L_list),1))[Ecv_list==minEcv])
print("Lambda with minumn Ecv: ", Lmin_cv,"\nEcv: ",minEcv)

#20

w = train(Lmin_cv, X,Y)

Ein = error(w, X,Y)
Eout = error(w,X_test, Y_test)
print("Ein: {} \n Eout: {}".format(Ein, Eout))
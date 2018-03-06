# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:48:57 2017

@author: ipingou
"""

import numpy as np
import sys

def prediction(g,beta,X):
    predict = np.zeros(X.shape[0])
    for i,x in enumerate(X):
        for j,x_s in enumerate(train_x):
            predict[i] = predict[i]+beta[j]*get_kernel(g,x,x_s)
    return predict
    
def errorRate(g,beta, c):
    if c ==0:
        predict = prediction(g,beta, test_x)
        #predict =  np.dot(test_x, w)
        return sum(np.sign(predict)!=test_y)/test_x.shape[0]
    if c == 1:
        #predict =  np.dot(train_x, w)
        predict = prediction(g,beta, train_x)

        return sum(np.sign(predict)!=train_y)/train_x.shape[0]
    
def get_kernel(g,x1,x2):
    d = x1-x2
    return np.exp(-1*g*(np.dot(d.T,d)))

    
def kernel(X, g):
    kernel = np.zeros([train_x.shape[0],train_x.shape[0]])
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[0]):
            #d = X[i]-X[j] 
            #kernel[i,j] = np.exp(-1*g*(np.dot(d.T,d)))
            kernel[i,j] = get_kernel(g,X[i],X[j])
    return kernel
    


input_train=sys.argv[1]
data =  np.loadtxt(input_train)
#data = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw2\\hw2_lssvm_all.dat")
train = data[0:400]
test = data[400:500]
train_x = train[:,0:10].copy()
train_x = np.insert(train_x,0,1,axis=1)
train_y = train[:,10].copy()
test_x = test[:,0:10].copy()
test_x = np.insert(test_x,0,1,axis=1)
test_y = test[:,10].copy()

Gamma = np.array([32,2,0.125])
Lambda =  np.array([0.001,1,1000])
C = 1/Lambda
epsilon = 0.5

result = []
for g in Gamma:
    kernelg = kernel(train_x, g)
    for l in Lambda:
        temp = l*(np.identity(train_x.shape[0]))+kernelg
        temp_inv = np.linalg.inv(temp)
        beta = np.dot(temp_inv, train_y)
        #w= np.dot(train_x.T,beta)
        ein = errorRate(g,beta,1)
        eout = errorRate(g,beta,0)
        result.append([g,l,beta,ein,eout])
        
for r in result:
    print("Gamma: ",r[0]," Lambda: ",r[1]," Ein: ",r[3])
   
#for r in result:
 #   print("Gamma: ",r[0]," Lambda: ",r[1]," Eout: ",r[4])
        


# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:26:23 2017

@author: ipingou
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:41:27 2017

@author: ipingou
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:48:57 2017

@author: ipingou
"""

import numpy as np
import random
from sklearn import datasets, linear_model

#import sys
#input_train=sys.argv[1]


#def return_w(a):
def prediction(g,beta,X,train_xxx):
    #predict n 個數字
    predict = np.zeros(X.shape[0])
    for i,x in enumerate(X):
        for j,x_s in enumerate(train_xxx):
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
    return np.dot(x1.T,x2)

    
def kernel(X, g):
    kernel = np.zeros([train_x.shape[0],train_x.shape[0]])
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[0]):
            #d = X[i]-X[j] 
            #kernel[i,j] = np.exp(-1*g*(np.dot(d.T,d)))
            kernel[i,j] = get_kernel(g,X[i],X[j])
    return kernel
    



data = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw2\\hw2_lssvm_all.dat")
train = data[0:400]
test = data[400:500]
train_x = train[:,0:10].copy()
train_x = np.insert(train_x,0,1,axis=1)
train_y = train[:,10].copy()
test_x = test[:,0:10].copy()
test_x = np.insert(test_x,0,1,axis=1)
test_y = test[:,10].copy()

#Gamma = np.array([32,2,0.125])
Lambda =  np.array([0.01, 0.1, 1, 10, 100])
#C = 1/Lambda
#epsilon = 0.5
T = 200


def bootstrap():
    random.seed()
    a = []
    k = random.randint(0,399)
    a.append(k)
    X = train_x[k]
    Y = train_y[k]
    for i in range(400-1):

        k = random.randint(0,399)
        a.append(k)
        X = np.vstack((X,train_x[k]))
        Y = np.vstack((Y,train_y[k]))     
    return (X,Y)

result = []

for l in Lambda:
    predict_in = np.zeros(train_x.shape[0])
    predict_out = np.zeros(test_x.shape[0])
    for i in range(T):
        print(i)
        train_xx,train_yy = bootstrap()
        #kernelg = kernel(train_xx,1)
        temp =l*np.identity(train_xx.shape[1])+ np.dot(train_xx.T,train_xx)
        temp_inv = np.linalg.inv(temp)
        temp_inv_xt = np.dot(temp_inv,train_xx.T)
        beta = np.dot(temp_inv_xt, train_yy)

        predict_g_in = np.sign(np.dot(train_x,beta)).flatten()
        print(np.dot(train_x,beta))
        predict_in = predict_in+predict_g_in
        predict_g_out = np.sign(np.dot(test_x,beta)).flatten()
        predict_out = predict_out+predict_g_out
    ein = sum(np.sign(predict_in)!=train_y)/train_x.shape[0]
    eout = sum(np.sign(predict_out)!= test_y)/test_x.shape[0]
    result.append([l,ein,eout])
    
        
        train_xx,train_yy = (train_x,train_y)

        train_xx,train_yy = bootstrap()

        predict_in = np.zeros(train_x.shape[0])
        predict_out = np.zeros(test_x.shape[0])

        l=0
        #train_xx,train_yy = (train_x,train_y)
        #kernelg = kernel(train_xx,1)
        temp =l*np.identity(train_xx.shape[1])+ np.dot(train_xx.T,train_xx)
        temp_inv = np.linalg.inv(temp)
        temp_inv_xt = np.dot(temp_inv,train_xx.T)
        beta = np.dot(temp_inv_xt, train_yy)

        predict = np.dot(train_x,beta)
        predict_g_in = np.sign(np.dot(train_x,beta)).flatten()
        print(np.dot(train_x,beta))
        predict_in = predict_in+predict_g_in
        predict_g_out = np.sign(np.dot(test_x,beta)).flatten()
        predict_out = predict_out+predict_g_out
        ein = sum(np.sign(predict_in)!=train_y)/train_x.shape[0]
        eout = sum(np.sign(predict_out)!= test_y)/test_x.shape[0]
        print(ein)
        
a = []        
for i in range(2000000):
    random.seed()
    a.append(random.randint(0,399))

x = np.array(a)
unique, counts = np.unique(x, return_counts=True)

print (np.asarray((unique, counts)).T)   
    
for r in result:
    print("Lambda: ",r[0]," Ein: ",r[1])
    
for r in result:
    print("Lambda: ",r[0]," Eout: ",r[2])
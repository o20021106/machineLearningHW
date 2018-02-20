# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 14:17:50 2017

@author: ipingou
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 11:18:59 2017

@author: ipingou
"""
import sys
import numpy as np
import math
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
    predict = np.sign(sigmoid(np.dot(X,w))-0.5)  
    #print(predict)
    return (1/X.shape[0])*sum(predict!=Y) 
    
[X,Y]=preprocess(input_train)
[X_test, Y_test] = preprocess(input_test)

def sigmoid_element(x):
   return 1/(1+math.exp(-1*x)) 

sigmoid = np.vectorize(sigmoid_element)

def gradient(x,y,w):
    wt = np.transpose(w)
    return (sigmoid(-1*y*np.dot(wt,x))*-1*y*x).reshape(len(w),1)
    
def error(w,X,Y):
    predict = np.sign(sigmoid(np.dot(X,w))-0.5)  
    #print(predict)
    return (1/X.shape[0])*sum(predict!=Y) 
    
w = np.zeros((X.shape[1],1))
step=0.001
n=0
for i in range(2000):
    n = n%X.shape[0]
    x=np.transpose(X[n])
    y=Y[n]     
    w = w-step*(gradient(x,y,w))
    #print(i)
    n=n+1
    
print(w)

print("Ein: " ,error(w,X_test,Y_test))
print ("Eout: ",error(w,X,Y))

#[[ 0.01826899]
 #[-0.01308051]
 #[ 0.04072894]
 #[-0.03295698]
 #[ 0.01498363]
 #[-0.03691042]
 #[ 0.01232819]
 #[ 0.04791334]
 #[-0.02244958]
 #[ 0.02470544]
 #[ 0.06878235]
 #[ 0.01897378]
 #[-0.02032107]
 #[-0.00901469]
 #[ 0.04589259]
 #[ 0.05776824]
 #[ 0.06102487]
 #[-0.04756147]
 #[ 0.06035018]
 #[-0.01660574]
 #[-0.03509342]]
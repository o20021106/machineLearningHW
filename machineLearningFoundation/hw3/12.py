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
import numpy as np
import math
k="C:/Users/ipingou/OneDrive/文件/碩二上/ml/hw3/hw3_train.dat"
data = np.loadtxt(k)
features = data.shape[1]
size = data.shape[0]
X = data[:,0:features-1]
X = np.insert(X,0,1,axis=1)
Y = data[:,-1]
Y=Y.reshape(len(Y),1)

def sigmoid_element(x):
   return 1/(1+math.exp(-1*x)) 

sigmoid = np.vectorize(sigmoid_element)

def gradient(x,y,w):
    wt = np.transpose(w)
    return (sigmoid(-1*y*np.dot(wt,x))*-1*y*x).reshape(len(w),1)
    
w = np.zeros((X.shape[1],1))
step=0.001
n=0
for i in range(2000):
    n = n%X.shape[0]
    x=np.transpose(X[n])
    y=Y[n]     
    w = w-0.01*(gradient(x,y,w))
    print(i)
    n=n+1
    
print(w)

#[[-0.00698773]
# [-0.20193137]
# [ 0.26707086]
# [-0.3591812 ]
# [ 0.0504073 ]
# [-0.37893176]
# [ 0.01152034]
# [ 0.3299883 ]
# [-0.26013873]
# [ 0.13198126]
# [ 0.49109572]
# [ 0.08408571]
# [-0.25939453]
# [-0.17543689]
# [ 0.30097778]
# [ 0.40304549]
# [ 0.43073192]
# [-0.46940448]
# [ 0.4304353 ]
# [-0.21371971]
# [-0.37813314]]
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

def gradient(X,Y,w):
    Xt = np.transpose(X)    
    Xw = np.dot(X,w)
    sig = sigmoid(np.multiply(-1*Y,Xw))
    return (1/X.shape[0])*(np.dot(Xt,np.multiply(sig,-1*Y))) 
    
w = np.zeros((X.shape[1],1))
step=0.001
for i in range(2000):
    w = w-step*(gradient(X,Y,w))

print (w)




print("Ein: " ,error(w,X_test,Y_test))
print ("Eout: ",error(w,X,Y))
#[[ 0.01878417]
# [-0.01260595]
 #[ 0.04084862]
 #[-0.03266317]
 #[ 0.01502334]
 #[-0.03667437]
 #[ 0.01255934]
 #[ 0.04815065]
 #[-0.02206419]
 #[ 0.02479605]
 #[ 0.06899284]
 #[ 0.0193719 ]
 #[-0.01988549]
 #[-0.0087049 ]
 # 0.04605863]
 #[ 0.05793382]
 #[ 0.061218  ]
 #[-0.04720391]
 #[ 0.06070375]
 #[-0.01610907]
 #[-0.03484607]]
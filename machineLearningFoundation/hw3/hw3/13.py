# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:57:51 2017

@author: ipingou
"""
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
input_train=sys.argv[1]
input_test=sys.argv[2]
k="C:/Users/ipingou/OneDrive/文件/碩二上/ml/hw3/hw4_train.dat"
l="C:/Users/ipingou/OneDrive/文件/碩二上/ml/hw3/hw4_test.dat"

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
    
[X,Y]=preprocess(input_train)
[X_test, Y_test] = preprocess(input_test)

Xt = np.transpose(X)
I=np.identity(X.shape[1])
L =1.126
XtY=np.dot(Xt,Y)


w= np.dot(np.linalg.inv(np.dot(Xt,X)+L*I),XtY)

print("#13")
print(w)
print("Ein: " ,error(w,X_test,Y_test))
print ("Eout: ",error(w,X,Y))



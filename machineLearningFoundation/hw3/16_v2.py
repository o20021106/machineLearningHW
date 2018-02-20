# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:42:48 2017

@author: ipingou
"""

import numpy as np
import math
import matplotlib.pyplot as plt

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
    
[X,Y]=preprocess(k)
[X_test, Y_test] = preprocess(l)
[X_train,X_val] = np.split(X, np.array([120]))
[Y_train, Y_val] = np.split(Y, np.array([120]))


#16
L_list_log = list(range(-10,3))
L_list=np.power(10,L_list_log,dtype=float)

Xt = np.transpose(X_train)
I=np.identity(X_train.shape[1])
XtY=np.dot(Xt,Y_train)

Ein_list=[]
Eval_list=[]
for L in L_list:
    w= np.dot(np.linalg.inv(np.dot(Xt,X_train)+L*I),XtY)
    Ein_list.append(error(w,X_train,Y_train))
    Eval_list.append(error(w,X_val,Y_val))
    print(error(w,X_train,Y_train))
    
plt.plot(L_list_log,Ein_list,'ro')
plt.show

minEin = min(Ein_list)
Lmin_train=max((L_list.reshape(len(L_list),1))[Ein_list==minEin])
print(minEin)
Eout_min_Ein = error(np.dot(np.linalg.inv(np.dot(Xt,X_train)+Lmin_train*I),XtY),X_test,Y_test)
print("Eout corresponding to min Ein:",Eout_min_Ein,"\n#16lambda: ", Lmin_train)

#17

L_list_log = list(range(-10,3))
plt.plot(L_list_log,Eval_list,'go')
plt.show

minEval = min(Eval_list)
Lmin_val=max((L_list.reshape(len(L_list),1))[Eval_list==minEval])

Eout_min_Eval = error(np.dot(np.linalg.inv(np.dot(Xt,X_train)+Lmin_val*I),XtY),X_test,Y_test)
print("Eout corresponding to min Eval:",Eout_min_Eval,"\n#17lambda: ", Lmin_val)

#18
Xt_whole = np.transpose(X)
XtY_whole=np.dot(Xt_whole,Y)
I=np.identity(X.shape[1])
w_whole = np.dot(np.linalg.inv(np.dot(Xt_whole,X)+Lmin_val*I),XtY_whole)

Ein = error(w_whole, X, Y)
Eout =  error(w_whole, X_test, Y_test )

print("Ein: ",Ein,"\nEout: ", Eout)

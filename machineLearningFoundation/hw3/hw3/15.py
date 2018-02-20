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
    
[X,Y]=preprocess(input_train)
[X_test, Y_test] = preprocess(input_test)

Xt = np.transpose(X)
I=np.identity(X.shape[1])
XtY=np.dot(Xt,Y)

L_list_log = np.array(list(range(-10,3)))
L_list=np.power(10,L_list_log,dtype=float)

Ein_list=[]
Eout_list=[]
for L in L_list:
    w= np.dot(np.linalg.inv(np.dot(Xt,X)+L*I),XtY)
    Ein_list.append(error(w,X,Y))
    Eout_list.append(error(w,X_test,Y_test))
    print(error(w,X,Y))

plt.plot(L_list_log,Eout_list,'bo')
plt.show

minEout = min(Eout_list)
Lmin=max((L_list.reshape(len(L_list),1))[Eout_list==minEout])
print("Eout min:",minEout)
print("#15lambda: ",Lmin)


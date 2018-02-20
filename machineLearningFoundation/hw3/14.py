# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:30:53 2017

@author: ipingou
"""

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

Xt = np.transpose(X)
I=np.identity(X.shape[1])
L =1.126
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

plt.plot(L_list_log,Ein_list,'ro')
plt.show

minEin = min(Ein_list)
Lmin=max((L_list.reshape(len(L_list),1))[Ein_list==minEin])
Eout_min_Ein = error(np.dot(np.linalg.inv(np.dot(Xt,X)+Lmin*I),XtY),X_test,Y_test)
print(minEin)

print("Eout corresponding to min Ein:",Eout_min_Ein,"\n#14lambda: ", Lmin)


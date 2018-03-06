# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:22:06 2016

@author: ipingou
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:05:14 2016

@author: ipingou
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt

input_train=sys.argv[1]
input_test=sys.argv[2]
data = np.loadtxt(input_train)
data = np.insert(data,0,1,axis=1)
size = data.shape[0]
features = data.shape[1]-1
X = data[:,0:features]
Y= data[:,-1]

data_test = np.loadtxt(input_test)
data_test = np.insert(data_test,0,1,axis=1)
size_test = data_test.shape[0]
features_test = data_test.shape[1]-1
X_test =data_test[:,0:features_test]
Y_test= data_test[:,-1]

def PLA(X,Y,SEED):
    """ 
    t: the number of update
    w_pocket: the pocket hypothesis
    w_update: the last updated hypothesis
    
    PLA take X,Y and return the an hypothesis generated according to pocket PLA\
        -update whenever the last updated hypothesis encounter an misclassified\
         x.
        -update pocket if the updated hypothesis has and lower error rate
    """
    random.seed(SEED)    
    
    t = 0
    w_pocket = np.zeros(features)
    w_update=  np.zeros(features)
    while(t <50):  
        index = random.randrange(0, data.shape[0]-1)            
        if (predict(w_update,X[index,:])!=Y[index]):
            w_update=update(w_update,X[index,:],Y[index])
            t+=1
            if error_rate(w_update,X,Y)<error_rate(w_pocket,X,Y):
                w_pocket=w_update
            if(t>=50):
                break
    return w_pocket

def update(w,x,y):
    w_update = w+y*x
    return w_update
    
def predict(w,x):
    return 1 if w.dot(x)>0 else -1
    
def error_rate(w,X,Y):
    """
    calculate error rate according to w
    """
    multiply = X.dot(w)
    predict = [1 if x>0 else -1 for x in multiply ]
    error = predict!=Y
    return np.mean(error)
    
w_list=[]
error_on_test=[]
        
"""
run 2000 experiments
in each round i, shuffle the data according to seed i.
error_on_test keep a record of the error rate on test set using the hypothesis\
 generated from the training set using PLA
"""        
for i in range(2000):
    w= PLA(X,Y,i)
    w_list.append(w)
    error_on_test.append(error_rate(w,X_test, Y_test))
    
print("Average Error Rate: ", np.mean(error_on_test))

plt.hist(error_on_test)
plt.ylabel("Frequency")
plt.xlabel("Error Rate")
plt.show()
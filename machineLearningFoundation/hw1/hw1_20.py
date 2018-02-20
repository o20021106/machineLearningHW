# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:18:53 2016

@author: ipingou
"""

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

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


df = pd.read_table("C:\\Users\\ipingou\\OneDrive\\文件\\碩二上\\ml\\hw1\\hw1_18_train.dat",sep=' |\t', engine="python", names = ['x1','x2','x3','x4','y'])
df.insert(0, "x0", 1)
data = np.array(df)
size = data.shape[0]
features = data.shape[1]-1
df_test= pd.read_table("C:\\Users\\ipingou\\OneDrive\\文件\\碩二上\\ml\\hw1\\hw1_18_test.dat",sep=' |\t', engine="python", names = ['x1','x2','x3','x4','y'])
df_test.insert(0, "x0", 1)
data_test = np.array(df_test)
size_test = data_test.shape[0]
features_test = data_test.shape[1]-1
X_test =data_test[:,0:features_test]
Y_test= data_test[:,-1]
 

#w_list=[]
#mistake=[]
def PLA(X,Y):
    t = 0

    w_pocket = np.zeros(features)
    w_update=  np.zeros(features)
    while(t <100):
        for i in range(size):           
            if (predict(w_update,X[i,:])!=Y[i]):
                w_update=update(w_update,X[i,:],Y[i])
              #  w_list.append(w_update)
               # mistake.append(i)
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
    multiply = X.dot(w)
    predict = [1 if x>0 else -1 for x in multiply ]
    error = predict!=Y
    return np.mean(error)
    
w_list=[]
error_on_test=[]
        
for i in range(2000):
    #print("experiment", i)
    index = list(range(size))    
    SEED = i
    random.seed(SEED)
    random.shuffle(index)
    Data = data[index, :]
    X = Data[:,0:features]
    Y= Data[:,-1]
    w= PLA(X,Y)
    w_list.append(w)
    error_on_test.append(error_rate(w,X_test, Y_test))
    
print("Average Error Rate: ", np.mean(error_on_test))
#0.131286


plt.hist(error_on_test)
plt.ylabel("Frequency")
plt.xlabel("Error Rate")
plt.show()
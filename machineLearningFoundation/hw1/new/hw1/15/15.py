# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 21:26:03 2016

@author: ipingou
"""
import sys
import numpy as np
input_file=sys.argv[1]
data = np.loadtxt(input_file)
data = np.insert(data,0,1,axis=1)
size = data.shape[0]
features = data.shape[1]-1
X = data[:,0:features]
Y= data[:,-1]

def PLA(X,Y):
    """
    t is the number of updates
    all_correct is the number of not encountering any misclassification for a \
    given hypothesis
    last mistake is the index of x according to which w was last modified
    w is the coefficients for all the features 
    
    the function take X (features) and Y(labels), and loop until a particular\
    hypothesis correctly classifies all samples(that is all_correct equals size).
    
    the function test whether the prediction given x and w was the same with\
    the label, if not, use PLA to update, and set all_correct to zero, to start\
    counting all over again.  
    """
    t = 0    
    all_correct = 0
    index = 0
    frequency= np.zeros(size)
    w = np.zeros(features)   
    while( all_correct != size):
        all_correct +=1
        index = (index)%size
        if (predict(w,X[index,:])!=Y[index]):
            w=w+Y[index]*X[index,:]
            all_correct =0
            t= t+1
            frequency[index]=frequency[index]+1
        index+=1
    return t , frequency, w    
    
def predict(w,x):
    """
    predict takes w and x and calculate matrix multiplication of the two. If \
    the result was postive, return 1, else -1.
    """
    return 1 if w.dot(x)>0 else -1
    
up_date_times, frequency , w= PLA(X,Y)
print ("Number of updates: ", up_date_times)    

m = max(frequency)
print ("Index of most number of updates: ", [l for l,n in enumerate(frequency) if n==m] ) 
#numb or updates: 45
#last mistake index: 135
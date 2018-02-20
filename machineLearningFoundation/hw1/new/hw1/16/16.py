# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:34:41 2016

@author: ipingou
"""
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

input_file=sys.argv[1]
data = np.loadtxt(input_file)
data = np.insert(data,0,1,axis=1)
size = data.shape[0]
features = data.shape[1]-1

def PLA(X,Y):
    """
    t is the number of updates before halt.
    all_correct keep record of the number of correct classification for current\
    hypothesis.
    w is the hypothesis
    last_mistake is the index of the x according to which w was last modified.    
    
    PLA(X,Y) takes X(features) and Y(labels),and return t,  and w
    It loops untill a particular hypothesis correctly classify all samples \
    that is all_correct equals size. 
    
    """
    t = 0
    all_correct = 0
    w = np.zeros(features)

    while(all_correct <size):
        all_correct=0
        for i in range(size):
            all_correct+=1
            if (predict(w,X[i,:])!=Y[i]):
                w=update(w,X[i,:],Y[i])
                all_correct =0
                t= t+1
                last_mistake = i

    return t , last_mistake, w    
    
def update(w,x,y):
    w_update = w+y*x
    return w_update
    
def predict(w,x):
    return 1 if w.dot(x)>0 else -1
    
update_list=[]
mistake_list=[]
        
for i in range(2000):
    index = list(range(size))    
    SEED = i
    random.seed(SEED)
    random.shuffle(index)
    Data = data[index, :]
    X = Data[:,0:features]
    Y= Data[:,-1]
    update_num,last_mistake, w= PLA(X,Y)
    update_list.append(update_num)
    mistake_list.append(last_mistake)
    
print("Average number of updates before algorithm halts: ", np.mean(update_list))
#40.0225

plt.hist(update_list)
plt.ylabel("Frequency")
plt.xlabel("Number of Updates")
plt.show()

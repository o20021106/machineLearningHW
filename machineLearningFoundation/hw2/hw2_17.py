# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:19:42 2016

@author: ipingou
"""
import numpy as np
import random
import matplotlib.pyplot as plt
sample_size=10

#biger index in x means postive index越大代表數字越大

def pos_neg_ray():
    x_generator = []
    for _ in range(sample_size):
        x_generator.append(random.uniform(-1,1))
    X = np.array(x_generator)
    X.sort()
    def y_generator(x,p=0.8):
        ran=random.random()
        return -1*np.sign(x) if ran>=p else np.sign(x)
            
    Y=np.array([y_generator(x) for x in X ])
    
    def error_rate(X,Y,theta_position,S):
        """
        theta0--x1--theta1--x2--x3----------theta9--x10---theta10
        theta0--x1--theta1--x2--x3----------theta19--x20--theta20    
        """
        predict=np.array( [S*np.sign(i-theta_position-0.1) for i in range (1,sample_size+1)])
        return sum(predict!=Y)/len(Y)
        
    error_pos=[error_rate(X,Y,theta,1) for theta in range(sample_size+1)]
    error_neg=[error_rate(X,Y,theta,-1) for theta in range(sample_size+1)]
    
    Ein = min(np.min(error_pos),np.min(error_neg))
    error_pos_index =  [i for i, x in enumerate(error_pos)if x == Ein]    
    error_neg_index =  [i for i, x in enumerate(error_neg)if x == Ein]
    
    theta_position =random.randint(0, (len(error_pos_index)+len(error_neg_index)-1))
    
    S1,S2=[0,0]
    
    if len(error_pos_index)==0:
        index = error_neg_index[theta_position]                
        S2=1    
    elif theta_position/len(error_pos_index)<1:
        index = error_pos_index[theta_position]
        S1=1
    else:
        index = error_neg_index[theta_position%(len(error_pos_index))]
        S2=1
        """
        theta0--x0--theta1--x1--x2----------theta9--x9--theta10
        theta0--x0--theta1--x1--x2----------theta19--x19--theta20    
        """
    if index==sample_size:
        theta=(X[index-1]+1)/2
    elif index==0:
        theta=(-1+X[index])/2
    else:
        theta=(X[index-1]+X[index])/2
        
    Eout = S1*(0.8*(abs(theta)/2)+0.2*(1-(abs(theta)/2)))+\
           S2*(0.8*(1-(abs(theta)/2))+0.2*(abs(theta)/2))
    
    return [Ein, Eout]



error = [pos_neg_ray() for _ in range(5000)]
Ein=np.array(error)[:,0]
Eout=np.array(error)[:,1]
print("Average Ein", np.mean(Ein))
print("Average Eout", np.mean(Eout))
plt.hist(Ein,bins=9)
plt.ylabel("Frequency")
plt.xlabel("In-sample Error Rate")
plt.show()


plt.hist(Eout)
plt.ylabel("Frequency")
plt.xlabel("Out-sample Error Rate")
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:05:54 2016

@author: ipingou
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:46:14 2016

@author: ipingou
"""

import numpy as np
import random 
import sys

d_train = sys.argv[1]
d_test = sys.argv[2] 
data_train = np.loadtxt(d_train)
data_test = np.loadtxt(d_test)

#data_train = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二上\\ml\\hw2\\hw2_train.dat")
#data_test = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二上\\ml\\hw2\\hw2_test.dat")
X= data_train[:,0:data_train.shape[1]-1]
Y = data_train[:,-1]
X_test= data_test[:,0:data_test.shape[1]-1]
Y_test = data_test[:,-1]

def pos_neg_di(X, Y,i):
    
    TEMP =np.column_stack((X,Y))
    TEMP=TEMP[TEMP[:,0].argsort()]
    X=TEMP[:,0]
    Y=TEMP[:,1]
    sample_size = len(X)
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

    return [theta, Ein,i,S1,S2]

   
theta_ein = [pos_neg_di(X[:,i],Y,i) for i in range(X.shape[1])]

theta_ein2 = np.array(theta_ein)

#print(theta_ein2)

Ein = min(theta_ein2[:,1])
Ein_list = theta_ein2[:,1]==Ein
theta_list=theta_ein2[Ein_list]

theta_index=random.randint(0, theta_list.shape[0]-1)
theta_di=theta_list[theta_index]
theta=theta_di[0]
S=-1
if theta_di[3]==1:
    S = 1


print("theta:",theta,"\nEin:",Ein,"\nS:",S, "\ndimension:", theta_di[2]+1)
#theta: 1.6175
#Ein: 0.25
#S: -1
#theta_dimension: 4th dimension


def error_test(X,Y,theta,S):
    TEMP =np.column_stack((X,Y))
    TEMP=TEMP[TEMP[:,0].argsort()]
    X=TEMP[:,0]
    Y=TEMP[:,1]
    sample_size = len(X)
    tempt = X>theta
    predict=np.where(tempt, 1*S, -1*S)
    Eout = sum(predict!=Y)/sample_size
    
    return Eout


    
Eout = error_test(X_test[:,theta_di[2]],Y_test,theta,S)
print(Eout)
#0.355
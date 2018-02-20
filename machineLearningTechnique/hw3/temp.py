# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:59:00 2017

@author: ipingou
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  7 13:41:24 2017

@author: ipingou
"""

import numpy as np
import matplotlib.pyplot

train_ = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw3\\hw3_train.dat")


U=(np.ones(train_.shape[0])*(1/train_.shape[0])).reshape([train_.shape[0],1])
train = np.concatenate((train_,U),axis=1)

def decision_stump():
    et = 10000000
    Sign = 0
    Theta = 0
    Theta_value = 0
    Dimension = 50    
    for i in range(2):
        temp = train[train[:,i].argsort()]
        for theta in range(train.shape[0]):
            if theta == 0:
                theta_value = -100000
            else:
                theta_value = (temp[:,i][theta]+temp[:,i][theta-1])/2
            predict = np.sign(train[:,i]-theta_value)
            predict = np.array([ 1 if i == 0 else i for i in predict])
            error = np.where(predict == train[:,2],0,1)
            error_sum= np.dot(error,train[:,3])/sum(train[:,3])
            two_error = np.array([error_sum,1-error_sum])
            sign_min = np.argmin(two_error)
            if two_error[sign_min] < et: 
                Sign = np.where( sign_min == 0 ,1,-1)
                Dimension = i
                Theta = theta
                et = two_error[sign_min]
                Theta_value = theta_value


            """
                          
            predict1=np.ones(theta)
            predict2=-1*np.ones(train.shape[0]-theta)
            predict = np.concatenate((predict1,predict2))
            error = np.where(predict ==temp[:,2],0,1)
            error_sum= np.dot(error,temp[:,3])/sum(temp[:,3])
            two_error = np.array([error_sum,1-error_sum])
            sign_min = np.argmin(two_error)
            if two_error[sign_min] < et: 
                Sign = np.where( sign_min == 0 ,1,-1)
                Dimension = i
                Theta = theta
                et = two_error[sign_min]
                if theta ==0:
                    Theta_value = -100000
                else:
                    Theta_value = (temp[theta]+temp[theta-1])/2
                    
            """
                    
    diamond_t = ((1-et)/et)**.5   
    prediction = Sign*(np.sign(train[:,Dimension]-Theta_value))
    prediction = np.array([ 1 if i == 0 else i for i in prediction])
    prediction_error = prediction == train[:,2]
    U_update = [U/diamond_t if prediction == True else U*diamond_t for (U,prediction) in zip(train[:,3],prediction_error) ]
    
    #print(et)
    return (Sign, Dimension,Theta, et, U_update,Theta_value)

def G_predict(t,X):
    prediction = np.zeros(X.shape[0])
    #print(prediction.shape)
    for i in range(t):
        Sign, Dimension,Theta_value = g[i]
        gt_prediction = Sign*(np.sign(X[:,Dimension]-Theta_value))
        gt_prediction = np.array([ 1 if i == 0 else i for i in gt_prediction])
        #print(gt_prediction.shape)
        prediction = prediction + alpha[i]*gt_prediction
    #print(prediction)
    error_rate = sum(np.sign(prediction) != X[:,2])/X.shape[0] 
    return error_rate   

def gt_predict(t,X):
    Sign, Dimension,Theta_value = g[t]
    gt_prediction = Sign*(np.sign(X[:,Dimension]-Theta_value))
    gt_prediction = np.array([ 1 if i == 0 else i for i in gt_prediction])
    #print(gt_prediction.shape)
    error_rate = sum(np.sign(gt_prediction) != X[:,2])/X.shape[0] 
    
    return error_rate
                        
                                       
    

    
alpha = []
g = []
U=[]
U.append(train[:,3].copy())
#train[:,3]=1
for i in range(300):
    Sign, Dimension,Theta, et, U_update, Theta_value = decision_stump()
    
    g.append([Sign, Dimension, Theta_value])
    U.append(np.array(U_update))
    train[:,3] = U_update
    alpha_t = np.log(((1-et)/et)**.5)
    alpha.append(alpha_t)


#7
Ein_gt=[]
for i in range(300):
    Ein_gt.append(gt_predict(i,train))

y = Ein_gt
x = list(range(1,301))
matplotlib.pyplot.scatter(x,y,s=1)
matplotlib.pyplot.show()

print("Ein(g1): ",Ein_gt[0]," alpha1: ",alpha[0])

#8
#9
Ein_Gt=[]
for i in range(300):
    Ein_Gt.append(G_predict(i,train))

y = Ein_Gt
x = list(range(1,301))
matplotlib.pyplot.scatter(x,y,s=1)
matplotlib.pyplot.show()

print("Ein(G): ",Ein_Gt[299])

#10

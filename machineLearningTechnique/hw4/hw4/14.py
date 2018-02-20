# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:48:54 2017

@author: ipingou
"""

import numpy as np
import random
import matplotlib.pyplot
import sys

input_train=sys.argv[1]
train =  np.loadtxt(input_train)
input_test = sys.argv[2]
test =  np.loadtxt(input_test)
#train = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw4\\hw3_train.dat")
#test = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw4\\hw3_test.dat")
n=train.shape[0] 


data = np.zeros((30000*n,3))
for i in range(30000*n):
    index = random.randint(0,n-1)
    data[i] = train[index,:]

def decision_stump(Data):
    #kkkkk=0
    #print("data size", Data.shape[0])
    Impurity = 10000000000000
    Sign = 0
    Theta = 0
    Theta_value = 0
    Dimension = 50    
    for i in range(2):
        temp = Data[Data[:,i].argsort()]
        for theta in range(Data.shape[0]+1):
            
            for s in (1,-1):
                if theta == Data.shape[0]:
                    theta_value = 100000
                elif theta == 0:
                    theta_value = -100000
                else:
                    #print(theta)
                    theta_value = (temp[:,i][theta]+temp[:,i][theta-1])/2
                predict = np.sign(Data[:,i]-theta_value)
                predict = s*np.array([ 1 if i == 0 else i for i in predict])
                index =np.array([True if i == 1 else False for i in predict])
                group1 = Data[index].copy()
                group2 = Data[np.logical_not(index)].copy()
                n1 = group1.shape[0]
                n2 = group2.shape[0]
                '''
                if kkkkk==0:
                    kkkkk=2
                    print('theta',theta)
                    print('n1',n1)
                    print('n2',n2)
                '''
                gini1, gini2 = (0,0)
                if n1 !=0:
                    gini1 = 2*(sum(group1[:,2]==1)/n1)*(1-(sum(group1[:,2]==1)/n1))
                if n2 !=0:
                    gini2 = 2*(sum(group2[:,2]==1)/n2)*(1-(sum(group2[:,2]==1)/n2))
                impurity = n1*gini1+n2*gini2
                #print('impurtity',impurity)
                #print('Impurity',Impurity)
                if impurity < Impurity: 
                    Sign = s
                    Dimension = i
                    Theta = theta
                    Impurity = impurity
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
                    

    return (Sign, Dimension,Theta,Theta_value,Impurity)

def DecisionTree(Data):
    #print(Data.shape)
    #print(Data[0][1])
    count_x=0
    count_y=0
    for i in range(Data.shape[0]-1):
        if np.sum(Data[i][0:2]!=Data[i+1][0:2])!=0:
            break
        count_x = i+2
        
    for i in range(Data.shape[0]-1):
        if Data[i][2]!=Data[i+1][2]:
            break
        count_y = i+2
        
    if count_x == Data.shape[0] or count_y == Data.shape[0]:
        #if count_x == Data.shape[0]:
           # print("data",Data)
        value, count = np.unique(Data[:,2],return_counts=True)
        return value[np.argmax(count)]
    
    elif Data.shape[0]==1:
        return Data[0][2]
        
    else:
        Sign, Dimension,Theta,Theta_value,impurity = decision_stump(Data)
        b = (Sign,Dimension,Theta_value)
        #print(Dimension,Theta,impurity)
        predict = np.sign(Data[:,Dimension]-Theta_value)
        predict = np.array([ 1 if i == 0 else i for i in predict])
        predict = Sign*np.array([ 1 if i == 0 else i for i in predict])
        index =np.array([True if i == 1 else False for i in predict])
        group1 = Data[index].copy()
        group2 = Data[np.logical_not(index)].copy()
        
        subtree1 = DecisionTree(group1)
        subtree2 = DecisionTree(group2)
    
        return ([b,subtree1,subtree2])

#train a tree(list form) with trainning data
#tree=DecisionTree(train)

#given an observation, return prediction  
def prediction(Data,Tree):

    #if tree is leaf(1 or -1) then return this basic tree
    if Tree == 1 or Tree == -1:
        return Tree
    #else if tree not the base case, go to next node with braching criteria
    else:
        Sign,Dimension,Theta_value=Tree[0]
        predict = Sign*np.sign(Data[Dimension]-Theta_value)
        temp1, temp2 = (0,0)
        if predict == 1:
            temp1 = 1
        else:
            temp2 = 1
        return (temp1)*prediction(Data, Tree[1])+(temp2)*prediction(Data, Tree[2])
        
#error rate
def error_rate(X,Tree):
    prediction_each = []
    for i in X:
        prediction_each.append(prediction(i,Tree))
    return (prediction_each,sum(prediction_each!=X[:,2])/X.shape[0])
kkk = DecisionTree(train)
aaa,bbb=error_rate(train,kkk)






"""
#15
prediction_in=[]    
for i in train:
    prediction_in.append(prediction(i,tree))
Ein = sum(prediction_in!=train[:,2])/train.shape[0]

prediction_out=[]    
for i in test:
    prediction_out.append(prediction(i,tree))
Eout = sum(prediction_out!=test[:,2])/test.shape[0]

print("Ein", Ein, " Eout: ", Eout)
"""
trees=[]
Ein=[]
Ein_G = []
Eout_G = []

predict_train = np.zeros((train.shape[0],1))
predict_test = np.zeros((test.shape[0],1))

for i in range(30000):
    #print('i',i)
    if i%100 == 0:
        print(i)
    pre_train = []
    pre_test=[]
    index_start = i*train.shape[0]
    index_finish = (i+1)*train.shape[0]
    
    train_tree = data[index_start:index_finish,:]                
    

    #print(train_tree.shape)
    tree_train = DecisionTree(train_tree)
    trees.append(tree_train)

    predict_tr, ein = error_rate(train,tree_train)
    predict_te, eout = error_rate(test, tree_train)    
        
    predict_train = predict_train +np.array(predict_tr).reshape(train.shape[0],1)
    predict_test = predict_test +np.array(predict_te).reshape(test.shape[0],1)
    
    train_p = np.sign(predict_train)
    test_p = np.sign(predict_test)
    #print(sum(test_p !=test[:,2])/test.shape[0])
    
    Ein.append(ein)
    Ein_G.append(sum(train_p!=train[:,2].reshape(train.shape[0],1))/train.shape[0])
    Eout_G.append(sum(test_p!=test[:,2].reshape(test.shape[0],1))/test.shape[0])
    
#14

matplotlib.pyplot.plot(range(30000),Eout_G,'-')

    

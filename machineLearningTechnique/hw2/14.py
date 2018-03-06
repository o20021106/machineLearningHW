# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:40:45 2017

@author: ipingou
"""
import numpy as np
from sklearn.svm import SVR
import sys
input_train=sys.argv[1]
data =  np.loadtxt(input_train)

#data = np.loadtxt("C:\\Users\\ipingou\\OneDrive\\文件\\碩二下2\\ml\\hw2\\hw2_lssvm_all.dat")
train = data[0:400]
test = data[400:500]
train_x = train[:,0:10].copy()
train_x = np.insert(train_x,0,1,axis=1)
train_y = train[:,10].copy()
test_x = test[:,0:10].copy()
test_x = np.insert(test_x,0,1,axis=1)
test_y = test[:,10].copy()

Gamma = np.array([32,2,0.125])
C_ = [0.001,1,100]
epsilon = 0.5

result = []

for g in Gamma:
    for c in C_:
        clf = SVR(C = c, epsilon=0.5, kernel = "rbf", gamma = g)
        clf.fit(train_x,train_y)
        predict_train = clf.predict(train_x)
        predict_test = clf.predict(test_x)
        ein = sum(np.sign(predict_train)!=train_y)/train_x.shape[0]
        eout = sum(np.sign(predict_test)!=test_y)/test_x.shape[0]
        result.append([g,c,ein,eout])
        
#for r in result:
 #   print("Gamma: ",r[0]," Lambda: ",r[1]," Ein: ",r[2])
    
for r in result:
    print("Gamma: ",r[0]," Lambda: ",r[1]," Eout: ",r[3])

        

        
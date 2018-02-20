# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:31:51 2017

@author: ipingou
"""
import numpy as np
import pandas as pd
from sklearn import svm

train = pd.read_csv("D:\kaggle\\train2.csv", sep=",")
test = pd.read_csv("D:\kaggle\\test.csv", sep=",")

data = pd.concat([train,test])

#preprocessing

data = data.drop(['Name','PassengerId'],axis=1)
data = pd.get_dummies(data, columns =['Pclass','Sex','Ticket','Cabin','Embarked'], prefix=['Pclass','Sex','Ticket','Cabin','Embarked'], prefix_sep = '_' )
mean_col = data.mean(axis=0)
for label in ['Age','Fare']:
    data[label] = data[label].replace('NaN', mean_col[label])

train_y = data.iloc[0:891,4]
x = data.drop('Survived',axis=1)
train_x = x.iloc[0:891, :]
test_x = x.iloc[891:,:]
    
#-----train
clf = svm.SVC(C=1000, kernel='rbf', gamma = 0.125)

clf.fit(train_x,train_y)
error_rate = sum(clf.predict(train_x)!=train_y)/len(train_y)
print(error_rate)

#-------test

predict_test = clf.predict(test_x)
pId_test = np.array(test.loc[:,'PassengerId'])

np.savetxt("D:\kaggle\\predict.csv", predict_test, delimiter=",")

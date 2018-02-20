# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:12:04 2017

@author: ipingou
"""

import numpy as np
import pandas as pd
from sklearn import svm
import glob
import scipy.sparse as sparse
from scipy.io import mmwrite
import timeit



test_files = glob.glob("D:\\ML2017\\datasets.tar\\datasets\\release1\\test_half_*")
train_files = glob.glob("D:\\ML2017\\datasets.tar\\datasets\\release1\\train_*")

max_feature = 0
#max_feature = 136

for train in train_files:
    print(train)
    with open(train) as f:
        for line in f:
            record = line.rstrip().split(' ')
            max_temp = int(max(record[4:]))
            max_feature = max(max_feature, max_temp)

max_feature_test = 0            
for train in test_files:
    print(train)
    with open(train) as f:
        for line in f:
            record = line.rstrip().split(' ')
            max_temp = int(max(record[3:]))
            max_feature_test = max(max_feature_test, max_temp)
            
max_feature = max(max_feature, max_feature_test)



with open(train_files[0]) as f:
   lines= f.readlines()
   features_matrix = np.zeros([len(lines),137])
   label = np.zeros(len(lines))
   time_and_ad_id = np.zeros([len(lines),2])
   for index, line in enumerate(lines):
       if index % 10000 == 0:
           print(index)
       record = line.rstrip().split(' ')
       label[index] = record[2]
       time_and_ad_id[index, :] = [int(record[0]),int(record[1][3:])]
       for feature in record[4:]:
           features_matrix[index,int(feature)]=1

data = []
for file in train_files:                         
    with open(file) as f:
        lines= f.readlines()
        features_matrix = np.zeros([len(lines),136])
        label = np.zeros(len(lines))
        time_and_ad_id = np.zeros([len(lines),2])
        for index, line in enumerate(lines):
            if index % 10000 == 0:
                print(index)
            record = line.rstrip().split(' ')
            label[index] = record[2]
            time_and_ad_id[index, :] = [int(record[0]),int(record[1][3:])]
            for feature in record[4:]:
                features_matrix[index,int(feature)-1]=1
        column_names = ['time','ad_id','label']+list(map(str,range(1,137)))
        data = pd.DataFrame(np.concatenate((time_and_ad_id,label.reshape(len(label),1),features_matrix),axis=1), columns= column_names)
        data = data.to_sparse()
        data.to_pickle(file+'trans')
      

for file in train_files:                         
    with open(file) as f:
        lines= f.readlines()
        matrix = np.zeros([len(lines),139])
        #label = np.zeros(len(lines))
        #time_and_ad_id = np.zeros([len(lines),2])
        for index, line in enumerate(lines):
            if index % 10000 == 0:
                print(index)
            record = line.rstrip().split(' ')
            matrix[index,2] = record[2]
            matrix[index, 0:2] = [int(record[0]),int(record[1][3:])]
            #time_and_ad_id[index, :] = [int(record[0]),int(record[1][3:])]
            for feature in record[4:]:
                matrix[index,int(feature)+2]=1
        column_names = ['time','ad_id','label']+list(map(str,range(1,137)))
        data = pd.DataFrame(matrix, columns= column_names)
        data = data.to_sparse()

def get_matrix(file,flag):
    if flag == 'train':
        with open(file) as f:
            lines= f.readlines()
            matrix = np.zeros([len(lines),139])
            #label = np.zeros(len(lines))
            #time_and_ad_id = np.zeros([len(lines),2])
            for index, line in enumerate(lines):
                if index % 10000 == 0:
                    print(index)
                record = line.rstrip().split(' ')
                matrix[index,2] = record[2]
                matrix[index, 0:2] = [int(record[0]),int(record[1][3:])]
                #time_and_ad_id[index, :] = [int(record[0]),int(record[1][3:])]
                for feature in record[4:]:
                    matrix[index,int(feature)+2]=1
            column_names = ['time','ad_id','label']+list(map(str,range(1,137)))
            data = pd.DataFrame(matrix, columns= column_names)
            #data = data.to_sparse()
    elif flag == 'test':
        with open(file) as f:
            lines= f.readlines()
            matrix = np.zeros([len(lines),138])
            #label = np.zeros(len(lines))
            #time_and_ad_id = np.zeros([len(lines),2])
            for index, line in enumerate(lines):
                if index % 10000 == 0:
                    print(index)
                record = line.rstrip().split(' ')
                #matrix[index,2] = record[2]
                matrix[index, 0:2] = [int(record[0]),int(record[1][3:])]
                #time_and_ad_id[index, :] = [int(record[0]),int(record[1][3:])]
                for feature in record[3:]:
                    matrix[index,int(feature)+1]=1
            column_names = ['time','ad_id']+list(map(str,range(1,137)))
            data = pd.DataFrame(matrix, columns= column_names)
            #print((sum(data.iloc[0,:])))
            #data = data.to_sparse()
    return data        

def get_sparse(matrix):
    return matrix.to_sparse()

test_path = 'D:\\ML2017\\datasets.tar\\datasets\\release1\\test_half_9'
a = get_matrix(train_files[0],'train')

x = a.to_sparse()
clf = svm.SVC()
start = timeit.default_timer()

clf.fit(x,a.label)

stop = timeit.default_timer()

print(stop - start )

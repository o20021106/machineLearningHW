# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:10:18 2017

@author: ipingou
"""

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

#this two lines get all the file names in testing and training data respectively
test_files = glob.glob("D:\\ML2017\\datasets.tar\\datasets\\release1\\test_half_*")
train_files = glob.glob("D:\\ML2017\\datasets.tar\\datasets\\release1\\train_*")

#get matrix accept a file path(file), and a flag('train' or 'test'), and returns
#a dataframe which contains binary data, transformed from 'file'. columns
#for trainning data are timestamp, ad id, label, and feature 1 to 136. The test
#data are the same, except there's no label column
def get_matrix(file,flag):
    #use flag to determin whether this is a trainning or testing data set
    if flag == 'train':
        with open(file) as f:
            #open file and read all the lines in the file into lines
            lines= f.readlines()
            #declare a matrix of zeros (n_row, 139)
            matrix = np.zeros([len(lines),139])
            #loop through each row to update the (n_row, 139) matrix
            for index, line in enumerate(lines):
                if index % 10000 == 0:
                    print(index)
                #preprcessing the line. Get rid of \n, and split data by ' '    
                record = line.rstrip().split(' ')
                #copy label
                matrix[index,2] = record[2]
                #copy timestamp and ad id
                matrix[index, 0:2] = [int(record[0]),int(record[1][3:])]
                #loop through the rest of the line, and update the matrix by the indices
                for feature in record[4:]:
                    matrix[index,int(feature)+2]=1
            #column names for pandas dataframe
            column_names = ['time','ad_id','label']+list(map(str,range(1,137)))
            #build a python data frame
            data = pd.DataFrame(matrix, columns= column_names)
    elif flag == 'test':
        with open(file) as f:
            lines= f.readlines()
            matrix = np.zeros([len(lines),138])
            for index, line in enumerate(lines):
                if index % 10000 == 0:
                    print(index)
                record = line.rstrip().split(' ')
                matrix[index, 0:2] = [int(record[0]),int(record[1][3:])]
                for feature in record[3:]:
                    matrix[index,int(feature)+1]=1
            column_names = ['time','ad_id']+list(map(str,range(1,137)))
            data = pd.DataFrame(matrix, columns= column_names)
    return data        

def get_sparse(matrix):
    return matrix.to_sparse()

data = get_matrix(train_files[0],'train')

"""
#this block contains codes to find the dimension of the data(and store it in 
#max_feature)

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
"""


import numpy as np
import random

import sys
input_train=sys.argv[1]
data = np.loadtxt(input_train)


#def return_w(a):
def prediction(g,beta,X,a,c):
    #predict n 個數字
           
    predict = np.zeros(X.shape[0])
    if c == 0:
        for i,x in enumerate(X):
            for j,x_s in enumerate(a):
                #predict[i] = predict[i]+beta[j]*get_kernel(g,x,x_s)
                predict[i] = predict[i]+beta[j]*Kernel_all[i,x_s]
        return predict
    else:
        for i,x in enumerate(X):
            for j,x_s in enumerate(a):
                #predict[i] = predict[i]+beta[j]*get_kernel(g,x,x_s)
                predict[i] = predict[i]+beta[j]*Kernel_all[i+400,x_s]
        return predict
    
def get_kernel(g,x1,x2):
    return np.dot(x1.T,x2)

    
def kernel(X, g):
    kernel = np.zeros([X.shape[0],X.shape[0]])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            #d = X[i]-X[j] 
            #kernel[i,j] = np.exp(-1*g*(np.dot(d.T,d)))
            kernel[i,j] = get_kernel(g,X[i],X[j])
    return kernel

train = data[0:400]
test = data[400:500]
train_x = train[:,0:10].copy()
train_x = np.insert(train_x,0,1,axis=1)
train_y = train[:,10].copy()
test_x = test[:,0:10].copy()
test_x = np.insert(test_x,0,1,axis=1)
test_y = test[:,10].copy()

#Gamma = np.array([32,2,0.125])
Lambda =  np.array([0.01, 0.1, 1, 10, 100])
#C = 1/Lambda
#epsilon = 0.5
T = 200


def bootstrap():
    random.seed()
    a = []
    k = random.randint(0,399)
    a.append(k)
    #X = train_x[k]
    Y = train_y[k]
    for i in range(400-1):
        k = random.randint(0,399)
        a.append(k)
       #X = np.vstack((X,train_x[k]))
        Y = np.vstack((Y,train_y[k]))     
    return (Y,a)

result = []
xxx=data[:,0:10]
all_x =np.insert(xxx,0,1,axis=1).copy()
Kernel_all = kernel(all_x,1)
def kernel_retrieve(a):
    kernel = np.zeros([len(a),len(a)])
    #print(a)
    for i,x1 in enumerate(a):
        for j,x2 in enumerate(a):
            kernel[i,j]= Kernel_all[x1,x2]
    return kernel
    

for l in Lambda:
    print(l)
    predict_in = np.zeros(train_x.shape[0])
    predict_out = np.zeros(test_x.shape[0])
    for i in range(T):
        train_yy,a = bootstrap()
        kernelg = kernel_retrieve(a)
        temp = l*(np.identity(len(a)))+kernelg
        temp_inv = np.linalg.inv(temp)
        beta = np.dot(temp_inv, train_yy)
        predict_g_in = np.sign(prediction(1,beta, train_x,a,0))
        predict_in = predict_in+predict_g_in
        predict_g_out = np.sign(prediction(1,beta, test_x,a,1))
        predict_out = predict_out+predict_g_out
    ein = sum(np.sign(predict_in)!=train_y)/train_x.shape[0]
    eout = sum(np.sign(predict_out)!= test_y)/test_x.shape[0]
    result.append([l,ein,eout])
   
    
#for r in result:
 #   print("Lambda: ",r[0]," Ein: ",r[1])
    
for r in result:
    print("Lambda: ",r[0]," Eout: ",r[2])
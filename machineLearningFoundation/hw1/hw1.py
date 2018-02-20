import pandas as pd
import numpy as np
df = pd.read_table("C:\\Users\\ipingou\\OneDrive\\文件\\碩二上\\ml\\hw1\\hw1_15_train.dat",sep=' |\t', engine="python", names = ['x1','x2','x3','x4','y'])
df.insert(0, "x0", 1)
data = np.array(df)
size = data.shape[0]
features = data.shape[1]-1
X = data[:,0:features]
Y= data[:,-1]

def PLA(X,Y):
    t = 0    
    all_correct = 0
    index = -1
    last_mistake =0
    w = np.zeros(features)   
    while( all_correct != size):
        all_correct +=1
        index = (index+1)%size
        if (predict(w,X[index,:])!=Y[index]):
            w=update(w,X[index,:],Y[index])
            all_correct =0
            t= t+1
            last_mistake = index         
    return t , last_mistake, w    
    
def update(w,x,y):
    w_update = w+y*x
    return w_update
    
def predict(w,x):
    return 1 if w.dot(x)>0 else -1
    
up_date_times, last_mistake_index , w= PLA(X,Y)
    
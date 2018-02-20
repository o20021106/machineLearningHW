import sys
import pandas as pd
import numpy as np
input_file=sys.argv[1]
df = pd.read_table(input_file,sep=' |\t', engine="python", names = ['x1','x2','x3','x4','y'])
df.insert(0, "x0", 1)
data = np.array(df)
size = data.shape[0]
features = data.shape[1]-1
X = data[:,0:features]
Y= data[:,-1]

def PLA(X,Y):
    """
    t is the number of updates
    all_correct is the number of not encountering any misclassification for a \
    given hypothesis
    last mistake is the index of x according to which w was last modified
    w is the coefficients for all the features 
    
    the function take X (features) and Y(labels), and loop until a particular\
    hypothesis correctly classifies all samples(that is all_correct equals size).
    
    the function test whether the prediction given x and w was the same with\
    the label, if not, use PLA to update, and set all_correct to zero, to start\
    counting all over again.  
    """
    t = 0    
    all_correct = 0
    index = 0
    last_mistake =0
    w = np.zeros(features)   
    while( all_correct != size):
        all_correct +=1
        index = (index)%size
        if (predict(w,X[index,:])!=Y[index]):
            w=update(w,X[index,:],Y[index])
            all_correct =0
            t= t+1
            last_mistake = index  
        index+=1
    return t , last_mistake, w    
    
def update(w,x,y):
    """
    update takes w,x,y and use PLA to update w
    """
    w_update = w+y*x
    return w_update
    
def predict(w,x):
    """
    predict takes w and x and calculate matrix multiplication of the two. If \
    the result was postive, return 1, else -1.
    """
    return 1 if w.dot(x)>0 else -1
    
up_date_times, last_mistake_index , w= PLA(X,Y)
print ("Number of updates: ", up_date_times)    
print ("Last: ", last_mistake_index) 
#45
#135
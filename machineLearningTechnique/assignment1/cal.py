import numpy as np 
from sklearn.svm import SVC

def K(x1,x2):
    return (2+np.dot(x1,x2))**2

def kernel2(X1,X2):
    gram_matrix = np.zeros((X1.shape[0],X2.shape[0]));
    for i, x in enumerate(X1):
        for j, y in enumerate(X2):
            gram_matrix[i,j] = K(x,y)
    return gram_matrix
            
X2 = np.array([[-3, -2], [3, -5], [3, -1],[5,-2],[9,-7],[9,1],[9,1]])
X = np.array([[1, 0], [0, 1], [0, -1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1, -1,-1,1,1,1,1])

clf = SVC(kernel = "linear")
clf.fit(X, y) 


clf2 = SVC(kernel = kernel2)
clf2.fit(X, y) 

clf3 = SVC(kernel = kernel2, degree = 2, coef0 = 2, gamma = 1)
clf3.fit(X, y) 


kkk=0
for i, j in enumerate(clf3.support_):
    print (i)
    aaa=K(X[j],X[1])*clf3.dual_coef_[0,i]
    kkk = kkk + aaa
    print (aaa)
    
print(-1-kkk)

kkk=0
for i, j in enumerate(clf3.support_):
    print (i)
    aaa=K(X[j],X[1])*clf3.dual_coef_[0,i]
    kkk = kkk + aaa
    print (aaa)
    
print(-1-kkk)


bbb=0
for i, j in enumerate(clf3.support_):
    print (j)
    print(clf3.dual_coef_[0,i])
    aaa=K(X[j],[100,23])*clf3.dual_coef_[0,i]
    bbb = bbb + aaa
    print (aaa)
    
print(-1.66+bbb)
print(bbb)
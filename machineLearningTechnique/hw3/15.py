# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:13:51 2017

@author: ipingou
"""

import numpy as np

import sys

input_train=sys.argv[1]
train =  np.loadtxt(input_train)
input_test = sys.argv[2]
test =  np.loadtxt(input_test)

 

   
def decision_stump(Data):
    #print("data size", Data.shape[0])
    Impurity = 100000000
    Sign = 0
    Theta = 0
    Theta_value = 0
    Dimension = 50    
    for i in range(2):
        temp = Data[Data[:,i].argsort()]
        for theta in range(Data.shape[0]-1):
            theta=theta+1
            for s in (1,-1):
                if theta == 0:
                    theta_value = -100000
                else:
                    theta_value = (temp[:,i][theta]+temp[:,i][theta-1])/2
                predict = np.sign(Data[:,i]-theta_value)
                predict = s*np.array([ 1 if i == 0 else i for i in predict])
                index =np.array([True if i == 1 else False for i in predict])
                group1 = Data[index].copy()
                group2 = Data[np.logical_not(index)].copy()
                n1 = group1.shape[0]
                n2 = group2.shape[0]
                gini1 = 2*(sum(group1[:,2]==1)/n1)*(1-(sum(group1[:,2]==1)/n1))
                gini2 = 2*(sum(group2[:,2]==1)/n2)*(1-(sum(group2[:,2]==1)/n2))
                impurity = n1*gini1+n2*gini2
                if impurity < Impurity: 
                    Sign = s
                    Dimension = i
                    Theta = theta
                    Impurity = impurity
                    Theta_value = theta_value
           

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
        if count_x == Data.shape[0]:
            print("data",Data)
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
tree=DecisionTree(train)

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
    return sum(prediction_each!=train[:,2])/train.shape[0]


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

#16
import copy
#class Tree is a node 
class Tree():
    "Generic tree node."
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.branching = None
        self.nodeId = None


#count is used to give nodes unique ID
count = 0

count_test =[]

#tree_list contains all the subtrees/nodes as well as the original tree constructed with list "tree"
tree_list = []
leafId = []
def buildTree(List):
    global leafId
    global count
    global count_test
    global tree_list
    if List == 1 or List == -1:
        node = Tree()
        node.nodeId = count
        count_test.append(count)
        count = count+1
        node.branching = List
        tree_list.append(node)

        return node 
    else:
        node = Tree()
        node.branching = List[0]
        node.nodeId = count
        #count_test.append(count)
        count = count+1
        node.left_child = buildTree(List[1])
        node.right_child=buildTree(List[2])
        tree_list.append(node)
        return node

#a is the tree constructed from the tree trained using trainning data in list form
a = buildTree(tree)

"""
#printing all subtrees/nodes of a
for i in tree_list:
    left_child_id = "leaf"
    right_child_id = "leaf"
    
    if (i.left_child != None) :
        left_child_id =i.left_child.nodeId
    if (i.right_child != None):
        right_child_id =i.right_child.nodeId
    
    print("ID:",i.nodeId," leftchild: ",left_child_id," right_child: ",right_child_id)    
    print("branching: ", i.branching)
"""    
        
#leafId stoers all leaf nodes
leafId = []
#if the node has no child, then it's a leaf, store the node's id into leaf, else trace through childs that's not none
def find_leafId(Tree):
    #print(Tree.left_child == None and Tree.right_child == None)
    if Tree.left_child == None and Tree.right_child == None:
        leafId.append(Tree.nodeId)
    elif Tree.left_child == None:
        find_leafId(Tree.right_child)
    elif Tree.right_child == None:
        find_leafId(Tree.left_child)     
    else:
        find_leafId(Tree.left_child)
        find_leafId(Tree.right_child)
#call find_leaFId to get all the leaaves' Id into leafId
find_leafId(a)

#given leaf ID and the tree, return a tree without leaf(id)
def delete_leaf(Tree,ID):
    #print(ID,Tree.nodeId,Tree.branching)
    
    # the first if prevent the programe to check children of leaves, which doesn't exist lol
    #if the program reach to a leaf and it's not the leaf we're looking for(ID), the program simply does nothing
    if not isinstance(Tree.branching,float):

        #if the node's left child is the leaf we're looking for, replace the node with it's right child
        if Tree.left_child.nodeId == ID: 
            Tree.branching = Tree.right_child.branching
            Tree.nodeId = Tree.right_child.nodeId
            Tree.left_child = Tree.right_child.left_child
            Tree.right_child = Tree.right_child.right_child
        #if the node's right child is the leaf we're looking for, replace the node with it's left child
        elif Tree.right_child.nodeId == ID:
            Tree.branching = Tree.left_child.branching
            Tree.nodeId = Tree.left_child.nodeId
            Tree.right_child = Tree.left_child.right_child
            Tree.left_child = Tree.left_child.left_child
        #if neither child is the node we're looking for, keep seraching 
        else:
            delete_leaf(Tree.left_child,ID)
            delete_leaf(Tree.right_child,ID)
       
        
#prune_tree is a list that store all the modified(pruened trees)        
prune_tree=[]    

#loop through all the leaf, and get a modified tree(deleting the specified leaf)
for i in leafId:
    tree_temp = copy.deepcopy(a)    
    delete_leaf(tree_temp,i)
    prune_tree.append(tree_temp)

#print tree
def print_tree(Tree):
    
    left_child_id = "leaf"
    right_child_id = "leaf"
    
    if (Tree.left_child != None) :
        left_child_id =Tree.left_child.nodeId
    if (Tree.right_child != None):
        right_child_id =Tree.right_child.nodeId
        
    if Tree.left_child == None and Tree.right_child == None:
        print("ID:",Tree.nodeId," leftchild: ",left_child_id," right_child: ",right_child_id," branching: ",Tree.branching)    
    elif Tree.left_child == None:
        print("ID:",Tree.nodeId," leftchild: ",left_child_id," right_child: ",right_child_id," branching: ",Tree.branching)    
        print_tree(Tree.right_child)
    elif Tree.right_child == None:
        print("ID:",Tree.nodeId," leftchild: ",left_child_id," right_child: ",right_child_id," branching: ",Tree.branching)    
        print_tree(Tree.left_child)
    else:
        print("ID:",Tree.nodeId," leftchild: ",left_child_id," right_child: ",right_child_id," branching: ",Tree.branching)    
        print_tree(Tree.left_child)
        print_tree(Tree.right_child)
"""      
#print all the pruned trees
for i in prune_tree:
    print_tree(i)
    print(" ")
"""
#given an observation and a tree, return the node that observation end up with according to branching criteria
#then return the nodeId, and the data 
def prediction_16(Data,Tree):
    #while not reaching a leaf, trace the tree according to bracnching criteria
    while not isinstance(Tree.branching, float):
        Sign,Dimension,Theta_value = Tree.branching
        predict = Sign*(np.sign(Data[Dimension]-Theta_value))
        
        #trace accroding to branching criteria
        if predict == 1:
            Tree = Tree.left_child
        else:
            Tree = Tree.right_child
    
    return [Tree.nodeId,Data[0],Data[1],Data[2]]
 
 #this function try's to get the value for the tree(because after pruning, 
 #and we need to calculate the new prediction values for each leaf. this is done
 #by classify each observation to a leaf, then calculate the majority(1 or -1)
#of y for each leaf with it's corresponding observations   
def update_value(Tree,X):
    #recal stores each data with it's corresponding leaf node id
    re_cal = []
    for Data in X:
        re_cal.append(prediction_16(Data,Tree))
    reCalculate = np.array(re_cal)
    #node_braching store each leaf id with it's corresponding prediction(-1 or +1)
    node_branching = []
    #loop through leaf node Id (using reCalculate), and calculate their corresponding prediction
    for nodeId in np.unique(reCalculate[:,0]):
        temp = reCalculate[reCalculate[:,0]==nodeId]
        value, count = np.unique(temp[:,3],return_counts = True)
        node_branching.append([nodeId, value[np.argmax(count)]])
    return node_branching

#this function try to update leaf i's prediction
#it search for the specific leaf node, than update it
def update_i(Tree,i,value):
    if Tree.left_child == None and Tree.right_child == None:
        if Tree.nodeId == i:
            Tree.branching = value
            #print("here ", Tree.nodeId,value)
    elif Tree.left_child == None:
        update_i(Tree.right_child,i,value)
    elif Tree.right_child == None:
        update_i(Tree.left_child,i,value)
    else:
        update_i(Tree.right_child,i,value)
        update_i(Tree.left_child,i,value)

#given a tree and node_branching(which stores leaf node id and their new prediction value)
#the function update the tree
def update(Tree,node_branching):
    for index, nodeId in enumerate(np.array(node_branching)[:,0]):
        #print("index value")
        #print(index)
        #print(node_branching[index][1])
        value = node_branching[index][1]
        #int(value)
        update_i(Tree,nodeId, value)
        
#loop through all the pruned trees and update their leaves' prediction
for prune_Tree in prune_tree:
    value=update_value(prune_Tree,train)
    update(prune_Tree,value)

#given and observation and a decision tree, the function return prediction of 
#the observation according to the decision tree
def prediction_tree(Data,Tree):

    if isinstance(Tree.branching,float):
        #print("here ", Tree)
        return Tree.branching
    else:
        Sign,Dimension,Theta_value=Tree.branching
        predict = Sign*np.sign(Data[Dimension]-Theta_value)
        temp1, temp2 = (0,0)
        if predict == 1:
            temp1 = 1
        else:
            temp2 = 1
        return (temp1)*prediction_tree(Data, Tree.left_child)+(temp2)*prediction_tree(Data, Tree.right_child)

#given a tree and a set of data, return prediction error rate 
def error_rate_tree(X,Tree):        
    prediction_16_list = []
    for i in X:
        prediction_16_list.append(prediction_tree(i,Tree))
    return sum(prediction_16_list!=X[:,2])/X.shape[0]


#loop trough all the prune tree and calculated all of the Ein
Ein_16 = []
for prune_Tree in prune_tree:
    #print(prune_Tree.nodeId)
    Ein_16.append(error_rate_tree(train,prune_Tree))
    
#loop trough all the prune tree and calculated all of the Eout
Eout_16 = []
for prune_Tree in prune_tree:
    #print(prune_Tree.nodeId)
    Eout_16.append(error_rate_tree(test,prune_Tree))
    
#print("min Ein: ", np.min(Ein_16), " min Eout: ",np.min(Eout_16))
    



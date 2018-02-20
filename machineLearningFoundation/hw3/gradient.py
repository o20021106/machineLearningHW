# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:58:39 2016

@author: ipingou
"""
import numpy as np
import math
def gradient_result(point):
    u=point[0]
    v=point[1]
    gradient = np.array([math.exp(u)+v*math.exp(u*v)+2*u-2*v-3,2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2])
    return np.array([u,v])-0.01*gradient
  
point=np.array([0,0])
for i in range(4000):
   point=gradient_result(point);
   print("\n\n\nu",i+1," = ",point[0],"\nv",i+1," = ",point[1])
    
    
def value(point):
    u=point[0]
    v=point[1]
    return math.exp(u)+math.exp(2*v)+math.exp(u*v)+u**2-2*u*v+2*(v**2)-3*u-2*v
    
def gradient(point):
    u=point[0]
    v=point[1]
    gradient = np.array([math.exp(u)+v*math.exp(u*v)+2*u-2*v-3,2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2])
    return gradient
    
def hessian(point):
    u=point[0]
    v=point[1]
    hessian=np.array([[math.exp(u)+(v**2)*math.exp(u)+2,math.exp(u*v)+u*v*math.exp(u*v)-2],[math.exp(u*v)+u*v*math.exp(u*v)-2,4*math.exp(2*v)+(u**2)*math.exp(u*v)+4]])
    return hessian
    
def newton(point):
    newton = -1*np.dot(np.linalg.inv(hessian(point)),np.transpose(gradient(point)))
    return newton
    
point=np.array([0,0])
for i in range(5):
    print("gradeint:", gradient(point))
    print("hessian:\n",hessian(point))
    print("hessian invers:\n",np.linalg.inv(hessian(point)))
    print("newton direction:", newton(point))
    point = point+newton(point)
    print("x               : ",point)
    print("\n\n")
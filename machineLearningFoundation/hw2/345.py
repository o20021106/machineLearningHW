# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 23:38:49 2016

@author: ipingou
"""
import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16])
import math
from decimal import Decimal

def ovc(n):
    return ((8/n)*math.log(4*((2*n)**50)/0.05))**.5
    
def vvc(n):
    return ((16/n)*math.log(2*((n)**50)/(0.05**.5)))**.5
    
def rpb(n):
    return ((2/n)*math.log(2*n*(n**50)))**.5+((2/n)*math.log(1/0.05))**.5+1/n

def pvdb(n):
    return ((1/n)*(2*0.05+math.log(6*(2*n)**50/0.05)))**.5
    
def devroye(n):
    return ((1/(2*n))*(4*(0.05)*(1.05)+math.log((n**100))+math.log((4/0.05))))**.5
  

a = []
b = []
c = []
d = []
e = []

  
for i in range(1,10001):
    a.append(ovc(i)) 
    b.append(vvc(i))
    c.append(rpb(i))
    d.append(pvdb(i))
    e.append(devroye(i))
    
    




import matplotlib.pyplot as plt
import numpy as np
#from mod import plotsomefunction
#from diffrentmod import plotsomeotherfunction

def plotsomefunction(ax):

    return ax.plot(range(1,10000), a, color="red")

def plotsomeotherfunction(ax):

    return ax.plot(range(1,10000),b,color="orange")


def plotsomeotherfunction2(ax):

    return ax.plot(range(1,10000),c,color="yellow")
    
def plotsomeotherfunction3(ax):

    return ax.plot(range(1,10000),d,color="green")
    
def plotsomeotherfunction4(ax):

    return ax.plot(range(1,10000),e,color="blue")
    

fig, ax = plt.subplots(1,1)
l1 = plotsomefunction(ax)
l2 = plotsomeotherfunction(ax)
l3 = plotsomeotherfunction2(ax)
l4 = plotsomeotherfunction3(ax)
l5 = plotsomeotherfunction4(ax)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:32:32 2021

@author: onepiece
"""
import random
q = dict()
state =(1, 1, 4, 4)
action = (1,1)
state=  tuple(state)
action = tuple(action)

q[(1, 1, 4, 4),(1,1)]=1
q[(1, 1, 4, 4),(0,1)]=0.5
# if q[state,action] == None:
   # print(0)
   
if (state,action) in q:
    print(1)
a= random.random()
actions = list()
actions.append(1)
actions.append(2)
actions2 = list()

r = random.randint(0,len(actions)-1)
# print(len(actions2)==0)
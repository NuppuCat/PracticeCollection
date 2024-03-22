# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:41:06 2020

@author: onepiece
"""

# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time
# Environment
env = gym.make("Taxi-v3")
# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode
# Q tables for rewards
# init should be -1,-100000 is too small
Q_reward = -1*numpy.ones((500,6))
# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE



p = 1000
for i in range(p):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action) 
        old_value = Q_reward[state, action]
        new_max = numpy.max(Q_reward[new_state])
        
        new_value =  old_value + alpha * (reward + gamma * new_max-old_value)
        Q_reward[state, action] = new_value
          
        
      
        state = new_state
    
    print(i)  
     
# print(numpy.max(Q_reward))


T = numpy.ones(10)
R = numpy.zeros(10)
for i in range(10):
    state = env.reset()
    tot_reward = 0
    t=0
    for t in range(1000):
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        # time.sleep(1)
        if done:
            print("Total reward %d" %tot_reward)
            break
    T[i] = t
    R[i] = tot_reward
    print("test %d time" %(i+1))
print("average Total reward %d ,average actions %d" %(numpy.average(R),numpy.average(T)))
    
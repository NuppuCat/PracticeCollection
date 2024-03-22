#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning

# ## OpenAI Gym Taxi Environment

# In[ ]:


# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3")
env.reset()
env.render()


# In[ ]:


# Run max 50 action steps and render output
for t in range(50):
    action = input('Give action number (0-5): ')
    new_state, reward, done, info = env.step(int(action))
    print('New state: %d Reward: %d Done: %d Info: %s' %(new_state,reward,done,info))
    env.render()


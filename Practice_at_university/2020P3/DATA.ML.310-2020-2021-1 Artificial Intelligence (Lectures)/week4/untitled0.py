# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:43:19 2021

@author: onepiece


 A = 0.03, B = 0.97, C = 0.22, D = 0.78, E = 0.99, and F = 0.01. Let the maximum error between consecutive iterations ε=0.01, and the discount factor γ=0.2. Round your answers to two decimal places.
A = 0.23, B = 0.77, C = 0.51, D = 0.49, E = 0.77, and F = 0.23.
A = 0.34, B = 0.66, C = 0.54, D = 0.46, E = 0.89, and F = 0.11.
"""
import numpy as np
states = [0,1,2]
actions = [0,1]
N_STATES = len(states)
N_ACTIONS = len(actions)
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards
P[1,0,1] = 1.0
P[0,0,1] = 1.0

# P[0,1,1] = 0.23
# P[0,1,2] = 0.77
# P[2,0,0] = 0.51
# P[2,0,2] = 0.49
# P[1,1,1] = 0.77
# P[1,1,2] = 0.23

# P[0,1,1] = 0.03
# P[0,1,2] = 0.97
# P[2,0,0] = 0.22
# P[2,0,2] = 0.78
# P[1,1,1] = 0.99
# P[1,1,2] = 0.01

P[0,1,1] = 0.34
P[0,1,2] = 0.66
P[2,0,0] = 0.54
P[2,0,2] = 0.46
P[1,1,1] = 0.89
P[1,1,2] = 0.11


R[0,0,1] = 1.0
R[0,1,1] = 2
R[0,1,2] = -1
R[1,0,1] = 1.0
R[1,1,1] = 2
R[1,1,2] = -1
R[2,0,0] = 1
R[2,0,2] = -1


gamma = 0.2
diata = 0.01
# initialize policy and value arbitrarily
policy = [0 for s in range(N_STATES)]
V = np.zeros(N_STATES)

is_value_changed = True
iterations = 0
while is_value_changed:
    is_value_changed = False
    iterations += 1
    # run value iteration for each state
    for s in range(N_STATES):
        V[s] = sum([P[s,policy[s],s1] * (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])
        # print "Run for state", s

    for s in range(N_STATES):
        q_best = V[s]
        # print "State", s, "q_best", q_best
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(N_STATES)])
            if q_sa > q_best:
                print ("State", s, ": q_sa", q_sa, "q_best", q_best)
                policy[s] = a
                q_best = q_sa
                is_value_changed = True

    print ("Iterations:", iterations)
    # print "Policy now", policy

print ("Final policy")
print (policy)
print (V)

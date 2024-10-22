# Parameters

import numpy as np
import random

machines = [1, 2, 3, 4]  # machines, it is just for downtime as it needs a list
n = len(machines)
# b = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]]).tolist()
b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).tolist()
# print(type(b))
# print(b)
parts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  #.tolist()
# print(type(parts))
ptypes = len(parts)
# print(type(ptypes))
null_part= [0,0,0]
# print(null_part)
p_sequence = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]).tolist()
# print(type(p_sequence))
B = np.array([15, 15, 15]).tolist()
# print(type(B))
Tp = np.array([4, 5, 4, 5]).tolist()
# print(type(Tp))
# Tl= np.array([1,1,1,1])
# Tu= np.array([1,1,1,1])
ng = 2
MTTR = np.array([20, 30, 25, 25]).tolist()
# print(type(MTTR))
MTBF = np.array([100, 120, 125, 150]).tolist()
# print(type(MTBF))
ms = np.array([1, 1, 1, 1]).tolist()  # either 0/1 machine on/off
# print(type(ms))
gs = np.array([0, 0, 0, 0]).tolist()  # either 0/1 depends on which machine it is assigned to
# print(type(gs))
n_wait = np.array([0, 0, 0, 0]).tolist()  # either 0/1 depends on if machine is waiting or not
# print(type(n_wait))
n_SB = np.array([0, 0, 0, 0]).tolist()  # either 0/1 depends on if machine is starved or blocked
# print(type(n_SB))
T = 2000
t=0
# mp = np.zeros([n, ptypes])  # part type on each machine
mp=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
# print(type(mp))
# if any(np.array_equal(sublist, [0, 0, 0]) for sublist in mp):
#     print("yes")
D= np.array([50,50,50]).tolist()
# print(type(D))
s_criticalIndx= 3
# m=[[]]*n
Tl= [2,2,2,2]
Tu= [2,2,2,2]
omega_d= np.array([1.2,1.3,1.1]).tolist()  # penalty for demand delay
# print(type(omega_d))
omega_s= np.array([1.2,1.3,1.1]).tolist() # surfeit penalty
# print(type(omega_s))
# if all(mp[0]==np.zeros(3)):
    # print('yes working')
# print(type(np.zeros([n,ptypes])))
import numpy as np

V = np.zeros((3, 4))
R = np.array([-2, 0, 3, -10])
P = np.array([[.9,.1, 0, 0],
              [.2,.5,.3, 0],
              [ 0,.3,.6,.1],
              [ 0, 0, 0, 0]])

def v(V, R, P, g):
    for k in range(V.shape[0]-1):
        for i in range(V.shape[1]):
            V[k+1][i] = sum((R[j] + g * V[k][j]) * P[i][j] for j in range(V.shape[1]))
    return V

V2 = np.zeros((20,3))
R2 = np.array([-.1, -.1, -1])
G = [[1],[0,2],[]]

def v2(V, R, G, g):
    for k in range(V.shape[0]-1):
        for i in range(V.shape[1]):
            if len(G[i]) != 0:
                V[k+1][i] = max((R[j] + g * V[k][j]) for j in G[i])
    return V

print(v2(V2, R2, G, 1))

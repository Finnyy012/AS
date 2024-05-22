import numpy as np


def value_iteration(max_iter, actions, reward, terminal, gamma, theta):
    values = np.zeros((max_iter + 1, reward.shape[0], reward.shape[1]))
    for k in range(max_iter):
        values[k+1] = np.max([a(reward) + gamma * a(values[k]) for a in actions], axis=0) * terminal
        delta = np.max(np.abs(values[k] - values[k + 1]))
        if delta < theta: return values[:k+1]
    return values


def policy(value, reward, gamma, terminal, symbols):
    pi = np.argmax([a(reward) + gamma * a(value) for a in actions], axis=0)
    return np.vectorize(lambda x: symbols[x])((pi + 1) * terminal.astype(int) - 1)


reward = np.array([[  -1,  -1, -1,   40],
                   [  -1,  -1, -10, -10],
                   [  -1,  -1,  -1,  -1],
                   [  10,  -2,  -1,  -1]])

terminal = np.ones((4,4))
terminal[3][0] = 0
terminal[0][3] = 0

actions = [lambda x: np.pad(x,((0,0),(1,0)), mode='edge')[ :  , :-1],
           lambda x: np.pad(x,((0,0),(0,1)), mode='edge')[ :  ,1:  ],
           lambda x: np.pad(x,((1,0),(0,0)), mode='edge')[ :-1, :  ],
           lambda x: np.pad(x,((0,1),(0,0)), mode='edge')[1:  , :  ]]

action_symbols = np.array(['<','>','^','v','X'])

max_iter = 10
theta = .01
gamma = 1
values = value_iteration(max_iter, actions, reward, terminal, gamma, theta)

print(values)
print()
print(policy(values[values.shape[0]-1], reward, gamma, terminal.astype(int), action_symbols))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def temporal_difference(pi, alpha, gamma, terminal, actions, reward, initial, episodes):
    V = np.zeros(pi.shape)
    for _ in range(episodes):
        S = initial
        while terminal[S[0]][S[1]]!=0:
            A = actions[pi[S[0]][S[1]]]
            S_prime = S + A
            R = reward[S_prime[0]][S_prime[1]]
            V[S[0]][S[1]] += alpha * (R + gamma * V[S_prime[0]][S_prime[1]] - V[S[0]][S[1]])
            S = S_prime
    return V


def SARSA(alpha, gamma, epsilon, terminal, actions, reward, initial, episodes):
    # Q = np.zeros((reward.shape[0],reward.shape[0],actions.shape[0]))
    Q = (np.random.rand(reward.shape[0],reward.shape[0],actions.shape[0]).T*terminal).T
    print(Q)
    S_heatmap = np.zeros(reward.shape)
    for i in range(episodes):
        S = initial
        S_heatmap[S[0]][S[1]] += 1
        A = epsilon_greedy(actions, Q[S[0]][S[1]], epsilon)
        while terminal[S[0]][S[1]]!=0:
            S_prime = S + actions[A]
            S_prime -= (S_prime == reward.shape).astype(int) - (S_prime == [-1, -1]).astype(int)
            R = reward[S_prime[0]][S_prime[1]]
            A_prime = epsilon_greedy(actions, Q[S_prime[0]][S_prime[1]], epsilon)
            Q[S[0]][S[1]][A] += alpha * (R + gamma * Q[S_prime[0]][S_prime[1]][A_prime] - Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
            S_heatmap *= .9
            S_heatmap[S[0]][S[1]] += 1
        # if i%10000==0:
        #     ax = sns.heatmap(S_heatmap, linewidth=0.5)
        #     plt.show()
    return Q, S_heatmap


def q_learning(alpha, gamma, epsilon, terminal, actions, reward, initial, episodes):
    Q = (np.random.rand(reward.shape[0], reward.shape[0], actions.shape[0]).T * terminal).T
    S_heatmap = np.zeros(reward.shape)
    for i in range(episodes):
        S = initial
        S_heatmap[S[0]][S[1]] += 1
        while terminal[S[0]][S[1]] != 0:
            A = epsilon_greedy(actions, Q[S[0]][S[1]], epsilon)
            S_prime = S + actions[A]
            S_prime -= (S_prime == reward.shape).astype(int) - (S_prime == [-1, -1]).astype(int)
            R = reward[S_prime[0]][S_prime[1]]
            Q[S[0]][S[1]][A] += alpha * (R + gamma * np.max(Q[S_prime[0]][S_prime[1]]) - Q[S[0]][S[1]][A])
            S = S_prime
            S_heatmap *= .9
            S_heatmap[S[0]][S[1]] += 1
        if i%3000==0:
            ax = sns.heatmap(S_heatmap, linewidth=0.5)
            plt.show()
    return Q

def epsilon_greedy(actions, Q, epsilon):
    p_epsilon = np.ones(actions.shape[0]) * epsilon / (actions.shape[0] - 1)
    p_epsilon[np.argmax(Q)] = 1 - epsilon
    return np.random.choice(actions.shape[0], p=p_epsilon)


def q_policy(Q, terminal):
    return np.apply_along_axis(np.argmax, 2, Q) * terminal


def prettify_policy(pi, symbols, terminal):
    return np.vectorize(lambda x: symbols[x])(pi + terminal.astype(int) - 1)


pi = np.array([[1, 1, 1, 0],
               [1, 2, 2, 2],
               [1, 2, 0, 0],
               [0, 2, 2, 0]])

reward = np.array([[  -1,  -1, -1,   40],
                   [  -1,  -1, -10, -10],
                   [  -1,  -1,  -1,  -1],
                   [  10,  -2,  -1,  -1]])

terminal = np.ones((4,4), dtype=int)
terminal[3][0] = 0
terminal[0][3] = 0

actions = np.array([[0,-1],[0,1],[-1,0],[1,0]])
action_symbols = np.array(['←','→','↑','↓','⊗'])

print(temporal_difference(pi, 1, 1, terminal, actions, reward, np.array((3,2)), 999))
print(temporal_difference(pi, 1, .5, terminal, actions, reward, np.array((3,2)), 999))

# Q, heatmap = SARSA(.99,.99,.1,terminal,actions,reward,np.array((3,2)),9999)
# print(Q)
# print(heatmap)
# ax = sns.heatmap(heatmap, linewidth=0.5)
# plt.show()

q = q_learning(.9,.9,.1,terminal,actions,reward,np.array((3,2)),100000)
print(q)
print(q_policy(q, terminal))
print(prettify_policy(q_policy(q, terminal), action_symbols, terminal))

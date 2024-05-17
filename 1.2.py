import numpy as np


def step(action, pos, reward):
    pos += action
    pos -= (pos == reward.shape).astype(int)
    pos += (pos == [-1,-1]).astype(int)
    return pos


def f_value(pos, actions, reward, value, terminal, γ):
    if not np.any(np.equal(terminal, pos).all(1)):
        v = np.zeros(actions.shape[0])
        for i in range(actions.shape[0]):
            a_res = step(actions[i], pos.copy(), reward)
            v[i] = reward[a_res[0]][a_res[1]] + γ * value[a_res[0]][a_res[1]]
        return np.max(v)
    else: return 0


def select_action(pos, policy):
    return policy[pos]


def act(policy, pos, reward, actions):
    pos = step(actions[policy[pos]], pos, reward)
    return pos


def value_iteration(max_iter, reward, actions, terminal, γ, θ):
    value = np.zeros((max_iter, reward.shape[0], reward.shape[1]))
    for k in range(value.shape[0]-1):
        for i in range(value.shape[1]):
            for j in range(value.shape[2]):
                value[k+1][i][j] = f_value(
                    np.array([i, j]),
                    actions,
                    reward,
                    value[k],
                    terminal,
                    γ
                )
        Δ = np.max(np.abs(value[k]-value[k+1]))
        if Δ < θ:
            return value[:k]
    return value


reward = np.array([[-1, -1, -1, 40],
                   [  -1,  -1, -10, -10],
                   [  -1,  -1,  -1,  -1],
                   [  10,  -2,  -1,  -1]])
policy = np.zeros((4,4))
terminal = np.array([[0,3], [3,0]])
pos = np.array([3,2])
actions = np.array([[ 1, 0],
                    [-1, 0],
                    [ 0, 1],
                    [ 0,-1]])

print(value_iteration(30, reward, actions, terminal, 1, 0.01))

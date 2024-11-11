#!/usr/bin/env python3
"""
Task 3
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """td lambtha with eligibility trace"""
    states, actions = Q.shape
    max_epsilon = epsilon

    def epsilon_greedy(epsilon, Qs):
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(Qs)
        else:
            return np.random.randint(0, 4)

    # evaluate episodes
    for i in range(episodes):
        E = np.zeros((states, actions))
        s_prev, _ = env.reset()
        action_prev = epsilon_greedy(epsilon, Q[s_prev])

        for j in range(max_steps):
            s, reward, terminated, truncated, _ = env.step(action_prev)
            action = epsilon_greedy(epsilon, Q[s])
            delta = reward + (gamma * Q[s, action]) - Q[s_prev, action_prev]
            E[s_prev, action_prev] += 1
            E = E * gamma * lambtha
            Q = Q + (alpha * delta * E)
            if terminated or truncated:
                break
            s_prev = s
            action_prev = action
        exp = np.exp(-epsilon_decay * i)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp
    return Q

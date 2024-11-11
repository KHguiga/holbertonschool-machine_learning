#!/usr/bin/env python3
"""
Task1
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """td lambtha with eligibility trace"""
    states = V.shape[0]
    # evaluate episodes
    for i in range(episodes):
        s, _ = env.reset()
        E = np.zeros(states)
        for j in range(max_steps):
            action = policy(s)
            s_new, reward, terminated, truncated, _ = env.step(action)
            delta = reward + (gamma * V[s_new]) - V[s]
            E[s] += 1
            V = V + alpha * delta * E
            E *= gamma * lambtha
            if terminated or truncated:
                break
            s = s_new
    return V

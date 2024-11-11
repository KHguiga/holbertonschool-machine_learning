#!/usr/bin/env python3
"""
Exercise with the Sarsa(λ) algorithm and the FrozenLakeEnv environment
By Ced
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    run 5000 episodes of sarsa(λ) algorithm
    """
    # Initialize eligibility traces, Q is given
    n_states, n_actions = Q.shape
    max_epsilon = epsilon

    for episode in range(episodes):
        """
        reset the environment and sample one episode
        player start upperleft
        Q is given
        """

        E = np.zeros((n_states, n_actions))
        done = False
        truncated = False

        # initialize state action
        state = env.reset()[0] # this gives  0
        action = get_action(state, Q, epsilon)

        while not (done or truncated):
            # observing next state and next action
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            if done or truncated:
                next_action = None
            else:
                next_action = get_action(next_state, Q, epsilon)

            target = reward + gamma * Q[next_state, next_action]
            actual = Q[state, action]
            delta = target - actual

            # Update eligibility trace for the current state
            # and Q values
            E[state, action] += 1  # Update eligibility
            E *= gamma * lambtha
            Q += alpha * delta * E  # update Qvalue

            # or?? but slower!
            # for s in range(n_states):
            #     for a in range(n_actions):
            #         Q[s, a] += alpha * delta * E[s, a]
            #         E[s, a] *= gamma * lambtha

            state, action = next_state, next_action

        # Decay epsilon after each episode
        exp = np.exp(-epsilon_decay * episode)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp
    
    return Q


def get_action(state, Q, epsilon):
    """
    Choose action using epsilon-greedy policy
    """
    n_actions = Q.shape[1]
    if np.random.rand() <= epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])

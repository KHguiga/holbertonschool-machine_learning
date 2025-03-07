#!/usr/bin/env python3
"""
Improved version of SARSA(λ)
"""
import numpy as np
import gym

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(λ) reinforcement learning algorithm.
    """
    # Validation
    if not (0 <= lambtha <= 1):
        raise ValueError("lambtha must be between 0 and 1")

    n_states, n_actions = Q.shape
    max_epsilon = epsilon
    E = np.zeros((n_states, n_actions))
    
    def get_action(epsilon, Qs):
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(Qs)
        else:
            # Ensure the number of actions matches your environment
            return np.random.randint(0, n_actions)
          
    for episode in range(episodes):
        E.fill(0)
        state, _ = env.reset()  # Assuming Gym environment
        action = get_action(epsilon, Q[state])
        done = truncated = False

        for j in range(max_steps):

            next_state, reward, done, truncated, _ = env.step(action)

            # Compute next action if game is over, no moves possible
            next_action = get_action(epsilon, Q[next_state])

            # Calcul optimisé
            delta = reward + (gamma * Q[next_state, next_action]) \
                - Q[state, action]

            # Update eligibility trace and Q table
            E[state, action] += 1            
            Q += alpha * delta * E
            E *= gamma * lambtha  # Update all eligibility traces

            if not (done or truncated):
                state, action = next_state, next_action
            else:
                break

        # Exponential decay of epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) \
            * np.exp(-epsilon_decay * episode)
    
    return Q

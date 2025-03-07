#!/usr/bin/env python3
"""
Improved version of SARSA(λ)
"""

import numpy as np

def sarsa_lambda(env, Q, lambda_val, episodes=5000, max_steps=100, alpha=0.1,
                 gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) reinforcement learning algorithm.

    Parameters:
    - env: The environment for the agent to interact with.
    - Q: The initial Q-table.
    - lambda_val: The lambda value for eligibility traces.
    - episodes: The number of episodes to run.
    - max_steps: The maximum number of steps per episode.
    - alpha: The learning rate.
    - gamma: The discount factor.
    - epsilon: The initial exploration rate.
    - min_epsilon: The minimum exploration rate.
    - epsilon_decay: Not used in this implementation; consider removing.

    Returns:
    - The updated Q-table after training.
    """

    # Validation
    if not (0 <= lambda_val <= 1):
        raise ValueError("lambda_val must be between 0 and 1")

    n_states, n_actions = Q.shape
    max_epsilon = epsilon
    eligibility_traces = np.zeros((n_states, n_actions))
    
    def get_action(epsilon, q_values):
        """
        Selects an action based on epsilon-greedy policy.
        
        Parameters:
        - epsilon: The exploration rate.
        - q_values: The Q-values for the current state.
        
        Returns:
        - The chosen action.
        """
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(q_values)
        else:
            # Ensure the number of actions matches your environment
            return np.random.randint(0, n_actions)
          
    initial_epsilon = epsilon  # Store initial epsilon for decay calculation

    for episode in range(episodes):
        eligibility_traces.fill(0)
        state, _ = env.reset()  # Assuming Gym environment
        action = get_action(epsilon, Q[state])
        done = truncated = False

        for step in range(max_steps):

            next_state, reward, done, truncated, _ = env.step(action)

            # Compute next action if game is over, no moves possible
            next_action = get_action(epsilon, Q[next_state])

            # Calculate TD error
            td_error = reward + (gamma * Q[next_state, next_action]) - Q[state, action]

            # Update eligibility trace and Q table
            eligibility_traces[state, action] += 1            
            Q += alpha * td_error * eligibility_traces
            eligibility_traces *= gamma * lambda_val  # Update all eligibility traces

            if not (done or truncated):
                state, action = next_state, next_action
            else:
                break

        # Exponential decay of epsilon
        epsilon = max(min_epsilon, initial_epsilon -(initial_epsilon - min_epsilon) * (episode / episodes))
    
    return Q

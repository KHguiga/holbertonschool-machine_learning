#!/usr/bin/env python3
"""
Improved version of SARSA(λ)
"""
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) reinforcement learning algorithm.
    """
    # Validation
    if not (0 <= lambtha <= 1):
        raise ValueError("lambtha must be between 0 and 1")

    n_states, n_actions = Q.shape
    initial_epsilon = epsilon
    E = np.zeros((n_states, n_actions))

    def epsilon_greedy(epsilon, Q, state):
        """Choisit une action en utilisant la politique epsilon-greedy"""
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(Q[state, :])
        else:
            return np.random.randint(0, Q.shape[1])

    for episode in range(episodes):
        E.fill(0)
        state = env.reset()[0]
        action = epsilon_greedy(epsilon, Q, state)
        steps = 0
        done = truncated = False

        while not (done or truncated) and steps < max_steps:
            steps += 1

            next_state, reward, done, truncated, _ = env.step(action)

            # compute next action if game is over, no moves possible
            next_action = epsilon_greedy(epsilon, Q, next_state) \
                if (not (done or truncated) and steps <= max_steps) else None

            # Calcul optimisé
            if not (done or truncated) and steps <= max_steps:
                delta = reward + (gamma * Q[next_state, next_action]) \
                    - Q[state, action]
            else:
                delta = reward - Q[state, action]

            # update eligibility trace and Q table
            E[state, action] += 1
            Q += alpha * delta * E
            E[state, action] *= gamma * lambtha

            if not (done or truncated) and steps <= max_steps:
                state, action = next_state, next_action
            else:
                break

        # Décroissance linéaire de epsilon
        epsilon = max(min_epsilon, initial_epsilon -
                      (initial_epsilon - min_epsilon) * (episode / episodes))
        # decroissance exponentielle de epsilon
        # epsilon = min_epsilon + (initial_epsilon - min_epsilon) \
        #     * np.exp(-epsilon_decay * episode)

    return Q

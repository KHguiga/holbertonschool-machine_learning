#!/usr/bin/env python3
"""
    Policy Gradient training
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    """

    scores = []
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        actions = []
        terminated = False
        truncated = False
        done = terminated or truncated

        while not done:
            if show_result is True and episode % 1000 == 0:
                env.render()

            # Get action and gradient from policy
            action, grad = policy_gradient(state, weight)
            # print(action)
            # print(grad)

            # Update environment
            state, reward, terminated, truncated, _ = env.step(action)

            # Update `done` flag after each step
            done = terminated or truncated

            # Expand state dimensions
            # state = state

            # Append to episode history
            grads.append(grad)
            rewards.append(reward)
            actions.append(action)
        for i in range(len(grads)):
            # Calculate rewards from this step forward
            reward = sum([R * gamma ** R for R in rewards[i:]])

            # Apply gradients
            weight += alpha * grads[i] * reward

        scores.append(sum(rewards))

        print('Episode: {} Score: {}'.format(episode, scores[episode]))

    return scores

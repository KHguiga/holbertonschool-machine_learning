#!/usr/bin/env python3
""" Policy function """
import numpy as np

# Our policy that maps state to action parameterized by w
def policy(matrix, weight):
	z = matrix.dot(weight)
	exp = np.exp(z)
	return exp/np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


def policy_gradient(state, weight):
	probs = policy(state, weight)
	action = np.random.choice(len(probs[0]), p=probs[0])
		
	# Compute gradient and save with reward in memory for our weight updates
	dsoftmax = softmax_grad(probs)[action,:]
	dlog = dsoftmax / probs[0, action]
	grad = state.T.dot(dlog[None, :])

	return action, grad

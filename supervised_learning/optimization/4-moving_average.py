#!/usr/bin/env python3
"""
   Mini-batch
"""
def moving_average(data, beta):
    avg = []
    prev = 0
    for i, d in enumerate(data):
        # weighted sum
        prev = beta * prev + (1 - beta) * d
        # correction
        correction = prev / (1 - (beta ** (i + 1)))
        avg.append(correction)
    return avg
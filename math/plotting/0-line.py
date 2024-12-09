#!/usr/bin/env python3
"""
line.py
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    line -- plotting Y as a line graph
    
    Return: None
    """
    
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color="red")
    plt.show()
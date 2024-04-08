#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.plot(x, y, 'm.')
# OR plt.scatter(x, y, c='m', marker='.')
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")
plt.show()
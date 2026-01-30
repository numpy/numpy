import matplotlib.pyplot as plt

import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 2, 3, 4, 5])
xx, yy = np.meshgrid(x, y)
plt.plot(xx, yy, marker='o', color='k', linestyle='none')

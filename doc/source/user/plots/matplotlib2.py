import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 5, 20)
y = np.linspace(0, 10, 20)
plt.plot(x, y, 'purple')  # line
plt.plot(x, y, 'o')      # dots
plt.show()

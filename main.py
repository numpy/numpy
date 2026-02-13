import numpy as np

a = np.ones((10, 10))
b = np.ones((10, 9))
np.asarray([a, b.T], dtype=object)

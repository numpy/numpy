import numpy as np

a: np.ndarray

np.require(a, requirements=1)  # E: No overload variant
np.require(a, requirements="TEST")  # E: incompatible type


"""Simple expression that should fail with mypy."""

import numpy as np

# Array creation routines checks
np.zeros("test")  # E: incompatible type
np.zeros()  # E: Too few arguments

np.ones("test")  # E: incompatible type
np.ones()  # E: Too few arguments

np.array(0, float, True)  # E: Too many positional

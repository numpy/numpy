import numpy as np

np.testing.bob  # E: Module has no attribute
np.bob  # E: Module has no attribute

# Stdlib modules in the namespace by accident
np.warnings  # E: Module has no attribute
np.sys  # E: Module has no attribute
np.os  # E: Module has no attribute
np.math  # E: Module has no attribute

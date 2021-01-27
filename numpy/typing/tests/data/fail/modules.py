import numpy as np

np.testing.bob  # E: Module has no attribute
np.bob  # E: Module has no attribute

# Stdlib modules in the namespace by accident
np.warnings  # E: Module has no attribute
np.sys  # E: Module has no attribute
np.os  # E: Module has no attribute
np.math  # E: Module has no attribute

np.__NUMPY_SETUP__  # E: Module has no attribute
np.__deprecated_attrs__  # E: Module has no attribute
np.__expired_functions__  # E: Module has no attribute

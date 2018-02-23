
from libc.stdint cimport uint64_t
cimport numpy as np

cdef np.ndarray seed_by_array(object seed, Py_ssize_t n)
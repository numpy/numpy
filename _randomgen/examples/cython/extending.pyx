import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from core_prng.common cimport prng_t
from core_prng.xoroshiro128 import Xoroshiro128

np.import_array()

def uniform_mean(Py_ssize_t N):
    cdef Py_ssize_t i
    cdef prng_t *rng
    cdef const char *anon_name = "CorePRNG"
    cdef double[::1] random_values
    cdef np.ndarray randoms

    x = Xoroshiro128()
    capsule = x._prng_capsule
    if not PyCapsule_IsValid(capsule, anon_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <prng_t *> PyCapsule_GetPointer(capsule, anon_name)
    random_values = np.empty(N)
    for i in range(N):
        random_values[i] = rng.next_double(rng.state)
    randoms = np.asarray(random_values)
    return randoms.mean()

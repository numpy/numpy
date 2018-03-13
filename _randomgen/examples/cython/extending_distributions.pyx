import numpy as np
cimport numpy as np
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from core_prng.common cimport *
from core_prng.xoroshiro128 import Xoroshiro128

@cython.boundscheck(False)
@cython.wraparound(False)
def normals_zig(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef prng_t *rng
    cdef const char *capsule_name = "CorePRNG"
    cdef double[::1] random_values

    x = Xoroshiro128()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <prng_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n)
    for i in range(n):
        random_values[i] = random_gauss_zig(rng)
    randoms = np.asarray(random_values)
    return randoms

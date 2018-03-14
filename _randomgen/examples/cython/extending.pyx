import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from randomgen.common cimport brng_t
from randomgen.xoroshiro128 import Xoroshiro128

np.import_array()

def uniform_mean(Py_ssize_t N):
    cdef Py_ssize_t i
    cdef brng_t *rng
    cdef const char *capsule_name = "BasicRNG"
    cdef double[::1] random_values
    cdef np.ndarray randoms

    x = Xoroshiro128()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <brng_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(N)
    for i in range(N):
        random_values[i] = rng.next_double(rng.state)
    randoms = np.asarray(random_values)
    return randoms.mean()

cdef uint32_t bounded_uint(uint32_t lb, uint32_t ub, brng_t *rng):
    cdef uint32_t mask, delta, val
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    val = rng.next_uint32(rng.state) & mask
    while val > delta:
        val = rng.next_uint32(rng.state) & mask

    return lb + val

@cython.boundscheck(False)
@cython.wraparound(False)
def bounded_uints(uint32_t lb, uint32_t ub, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef brng_t *rng
    cdef uint32_t[::1] out
    cdef const char *capsule_name = "BasicRNG"

    x = Xoroshiro128()
    out = np.empty(n, dtype=np.uint32)
    capsule = x.capsule
    
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <brng_t *>PyCapsule_GetPointer(capsule, capsule_name)

    for i in range(n):
        out[i] = bounded_uint(lb, ub, rng)
    return np.asarray(out)

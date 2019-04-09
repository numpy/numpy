#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random.randomgen.common cimport *
from numpy.random.randomgen.distributions cimport random_gauss_zig
from numpy.random.randomgen import Xoroshiro128


@cython.boundscheck(False)
@cython.wraparound(False)
def normals_zig(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef brng_t *rng
    cdef const char *capsule_name = "BasicRNG"
    cdef double[::1] random_values

    x = Xoroshiro128()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <brng_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n)
    for i in range(n):
        random_values[i] = random_gauss_zig(rng)
    randoms = np.asarray(random_values)
    return randoms


@cython.boundscheck(False)
@cython.wraparound(False)
def uniforms(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef brng_t *rng
    cdef const char *capsule_name = "BasicRNG"
    cdef double[::1] random_values

    x = Xoroshiro128()
    capsule = x.capsule
    # Optional check that the capsule if from a Basic RNG
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <brng_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n)
    for i in range(n):
        # Call the function
        random_values[i] = rng.next_double(rng.state)
    randoms = np.asarray(random_values)
    return randoms

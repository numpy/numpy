#!/usr/bin/env python
#cython: language_level=3
"""
This file shows how the distributions that are accessed through
distributions.pxd can be used Cython code.
"""
import numpy as np
cimport numpy as np
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random._bit_generator cimport bitgen_t
from numpy.random import PCG64


# cython example

@cython.boundscheck(False)
@cython.wraparound(False)
def uint16_uniforms(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double[::1] random_values

    x = PCG64()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='uint32')
    # Best practice is to release GIL and acquire the lock
    with x.lock, nogil:
        for i in range(n):
            random_values[i] = rng.next_uint32(rng.state)
    randoms = np.asarray(random_values)
    return randoms


@cython.boundscheck(False)
@cython.wraparound(False)
def uniforms(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double[::1] random_values

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')
    with x.lock, nogil:
        for i in range(n):
            # Call the function
            random_values[i] = rng.next_double(rng.state)
    randoms = np.asarray(random_values)
    return randoms

import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from common cimport *

from core_prng.splitmix64 import SplitMix64

np.import_array()

cdef class RandomGenerator:
    """
    Prototype Random Generator that consumes randoms from a CorePRNG class

    Parameters
    ----------
    prng : CorePRNG, optional
        Object exposing a PyCapsule containing state and function pointers

    Examples
    --------
    >>> from core_prng.generator import RandomGenerator
    >>> rg = RandomGenerator()
    >>> rg.random_integer()
    """
    cdef public object __core_prng
    cdef anon_func_state anon_rng_func_state
    cdef random_uint64_anon next_uint64
    cdef void *rng_state

    def __init__(self, prng=None):
        if prng is None:
            prng = SplitMix64()
        self.__core_prng = prng

        capsule = prng._anon_func_state
        cdef const char *anon_name = "Anon CorePRNG func_state"
        if not PyCapsule_IsValid(capsule, anon_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self.anon_rng_func_state = (<anon_func_state *>PyCapsule_GetPointer(capsule, anon_name))[0]
        self.next_uint64 = <random_uint64_anon>self.anon_rng_func_state.f
        self.rng_state = self.anon_rng_func_state.state

    def random_integer(self):
        return self.next_uint64(self.rng_state)

    def random_double(self):
        return (self.next_uint64(self.rng_state) >> 11) * (1.0 / 9007199254740992.0)

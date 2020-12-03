cimport numpy as np
IF UNAME_SYSNAME == 'OpenVMS':
    from stdint_fake cimport uint32_t, uint64_t
ELSE:
    from libc.stdint cimport uint32_t, uint64_t

cdef extern from "numpy/random/bitgen.h":
    struct bitgen:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    ctypedef bitgen bitgen_t

from numpy.random.bit_generator cimport BitGenerator, SeedSequence

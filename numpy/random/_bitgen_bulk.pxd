from libc.stdint cimport uint32_t, uint64_t

cdef extern from "numpy/random/bitgen_bulk.h":
    ctypedef struct bitgen_bulk_v1:
        uint32_t abi_version
        size_t struct_size
        uint64_t capabilities
        void (*fill_uint32)(void *state, size_t count, uint32_t *out) noexcept nogil
        void (*fill_uint64)(void *state, size_t count, uint64_t *out) noexcept nogil
        void (*fill_double)(void *state, size_t count, double *out) noexcept nogil

    enum:
        BITGEN_BULK_ABI_VERSION
        BITGEN_BULK_UINT32
        BITGEN_BULK_UINT64
        BITGEN_BULK_DOUBLE

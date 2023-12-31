#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "templ_common.h"

/*
 * Helper function taking the size input and growing it (based on min_grow).
 * The current scheme is a minimum growth and a general growth by 25%
 * overallocation.  This is then capped at 2**20 elements, as that propels us
 * in the range of large page sizes (so it is presumably more than enough).
 *
 * It further multiplies it with `itemsize` and ensures that all results fit
 * into an `npy_intp`.
 * Returns -1 if any overflow occurred or the result would not fit.
 * The user has to ensure the input is ssize_t but not negative.
 */
NPY_NO_EXPORT npy_intp
grow_size_and_multiply(npy_intp *size, npy_intp min_grow, npy_intp itemsize) {
    /* min_grow must be a power of two: */
    assert((min_grow & (min_grow - 1)) == 0);
    npy_uintp new_size = (npy_uintp)*size;
    npy_intp growth = *size >> 2;
    if (growth <= min_grow) {
        /* can never lead to overflow if we are using min_growth */
        new_size += min_grow;
    }
    else {
        if (growth > 1 << 20) {
            /* limit growth to order of MiB (even hugepages are not larger) */
            growth = 1 << 20;
        }
        new_size += growth + min_grow - 1;
        new_size &= ~min_grow;

        if (new_size > NPY_MAX_INTP) {
            return -1;
        }
    }
    *size = (npy_intp)new_size;
    npy_intp alloc_size;
    if (npy_mul_sizes_with_overflow(&alloc_size, (npy_intp)new_size, itemsize)) {
        return -1;
    }
    return alloc_size;
}


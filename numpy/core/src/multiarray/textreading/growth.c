#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "templ_common.h"

#include "textreading/growth.h"


/*
 * Helper function taking the size input and growing it (based on min_grow).
 * It further multiplies it with `itemsize` and ensures that all results fit
 * into an `npy_intp`.
 * Returns -1 if any overflow occurred or the result would not fit.
 * The user has to ensure the input is size_t (i.e. unsigned).
 */
npy_intp
grow_size_and_multiply(size_t *size, size_t min_grow, npy_intp itemsize) {
    /* min_grow must be a power of two: */
    assert((min_grow & (min_grow - 1)) == 0);
    size_t growth = *size >> 2;
    if (growth <= min_grow) {
        *size += min_grow;
    }
    else {
        *size += growth + min_grow - 1;
        *size &= ~min_grow;

        if (*size > NPY_MAX_INTP) {
            return -1;
        }
    }

    npy_intp res;
    if (npy_mul_with_overflow_intp(&res, (npy_intp)*size, itemsize)) {
        return -1;
    }
    return res;
}


#ifndef NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_

#include "numpy/npy_math.h"

/* Whether a float is in [min, max + 1); evaluates value more than once. */
#define NPY_FLOAT_IN_INT_RANGE(value, min, max) \
    (npy_isfinite(value) && (value) >= (double)(min) && \
            (value) < 2.0 * ((max) / 2 + 1))

/* Truncate toward zero; on failure return -1 and leave *out unchanged. */
#define NPY_DEFINE_DOUBLE_TO_INT(name, type, min, max) \
    static inline int \
    npy_double_to_##name(double value, type *out) \
    { \
        value = npy_trunc(value); \
        if (!NPY_FLOAT_IN_INT_RANGE(value, min, max)) { \
            return -1; \
        } \
        *out = (type)value; \
        return 0; \
    }

NPY_DEFINE_DOUBLE_TO_INT(int, int, INT_MIN, INT_MAX)
NPY_DEFINE_DOUBLE_TO_INT(intp, npy_intp, NPY_MIN_INTP, NPY_MAX_INTP)
NPY_DEFINE_DOUBLE_TO_INT(int64, npy_int64, NPY_MIN_INT64, NPY_MAX_INT64)
#undef NPY_DEFINE_DOUBLE_TO_INT

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_ */

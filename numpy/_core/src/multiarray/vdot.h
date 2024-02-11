#ifndef NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_

#include "common.h"

NPY_NO_EXPORT void
CFLOAT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CLONGDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
OBJECT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_ */

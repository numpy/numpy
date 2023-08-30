#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMMON_DTYPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMMON_DTYPE_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <numpy/ndarraytypes.h>
#include "dtypemeta.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2);

NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypeSequence(
        npy_intp length, PyArray_DTypeMeta **dtypes_in);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMMON_DTYPE_H_ */

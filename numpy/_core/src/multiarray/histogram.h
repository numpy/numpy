#ifndef NUMPY_CORE_SRC_MULTIARRAY_HISTOGRAM_H_
#define NUMPY_CORE_SRC_MULTIARRAY_HISTOGRAM_H_

#include "numpy/ndarraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT PyObject *
arr_histogram_uniform(PyObject *, PyObject *const *, Py_ssize_t, PyObject *);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HISTOGRAM_H_ */

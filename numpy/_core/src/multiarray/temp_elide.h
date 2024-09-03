#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

NPY_NO_EXPORT int
can_elide_temp_unary(PyArrayObject * m1);

NPY_NO_EXPORT int
try_binary_elide(PyObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_ */

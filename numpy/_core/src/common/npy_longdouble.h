#ifndef NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_

#include "npy_config.h"
#include "numpy/ndarraytypes.h"

/* Convert a npy_longdouble to a python `long` integer.
 *
 * Results are rounded towards zero.
 *
 * This performs the same task as PyLong_FromDouble, but for long doubles
 * which have a greater range.
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval);

/* Convert a python `long` integer to a npy_longdouble
 *
 * This performs the same task as PyLong_AsDouble, but for long doubles
 * which have a greater range.
 *
 * Returns -1 if an error occurs.
 */
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_ */

#ifndef __NPY_LONGDOUBLE_H
#define __NPY_LONGDOUBLE_H

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

#endif

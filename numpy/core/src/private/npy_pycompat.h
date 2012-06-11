#ifndef _NPY_PYCOMPAT_H_
#define _NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"

/*
 * Accessing items of ob_base
 */

#if (PY_VERSION_HEX < 0x02060000)
#define Py_TYPE(o)    (((PyObject*)(o))->ob_type)
#define Py_REFCNT(o)  (((PyObject*)(o))->ob_refcnt)
#define Py_SIZE(o)    (((PyVarObject*)(o))->ob_size)
#endif

/*
 * PyIndex_Check
 */
#if (PY_VERSION_HEX < 0x02050000)
#undef PyIndex_Check
#define PyIndex_Check(o)     0
#endif

#endif /* _NPY_COMPAT_H_ */

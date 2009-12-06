#ifndef _NPY_3KCOMPAT_H_
#define _NPY_3KCOMPAT_H_

#include <Python.h>
#include "npy_config.h"

/*
 * PyInt -> PyLong
 */

#if defined(NPY_PY3K)
/* Return True only if the long fits in a C long */
static NPY_INLINE int PyInt_Check(PyObject *op) {
    int overflow = 0;
    if (!PyLong_Check(op)) {
        return 0;
    }
    PyLong_AsLongAndOverflow(op, &overflow);
    return (overflow == 0);
}

#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AsLong
#define PyInt_AsSsize_t PyLong_AsSsize_t

/* NOTE:
 *
 * Since the PyLong type is very different from the fixed-range PyInt,
 * we don't define PyInt_Type -> PyLong_Type.
 */
#endif /* NPY_PY3K */

/*
 * PyString -> PyBytes
 */

#if defined(NPY_PY3K)
#define PyString_Type PyBytes_Type
#define PyString_Check PyBytes_Check

#define PyStringObject PyBytesObject

#define PyString_FromString PyBytes_FromString
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_FromFormat PyBytes_FromFormat
#define PyString_Concat PyBytes_Concat
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString
#define PyString_GET_SIZE PyBytes_GET_SIZE
#endif /* NPY_PY3K */

/*
 * Accessing items of ob_base
 */

#if (PY_VERSION_HEX < 0x02060000)
#define Py_TYPE(o)    (((PyObject*)(o))->ob_type)
#define Py_REFCNT(o)  (((PyObject*)(o))->ob_refcnt)
#define Py_SIZE(o)    (((PyVarObject*)(o))->ob_size)
#endif

/*
 * PyObject_Cmp
 */
#if defined(NPY_PY3K)
static NPY_INLINE int
PyObject_Cmp(PyObject *i1, PyObject *i2, int *cmp)
{
    int v;
    v = PyObject_RichCompareBool(i1, i2, Py_LT);
    if (v == 0) {
        *cmp = -1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    v = PyObject_RichCompareBool(i1, i2, Py_GT);
    if (v == 0) {
        *cmp = 1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    v = PyObject_RichCompareBool(i1, i2, Py_EQ);
    if (v == 0) {
        *cmp = 0;
        return 1;
    }
    else {
        *cmp = 0;
        return -1;
    }
}
#endif

#endif /* _NPY_3KCOMPAT_H_ */

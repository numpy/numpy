#ifndef _NPY_3KCOMPAT_H_
#define _NPY_3KCOMPAT_H_

#include "npy_config.h"

#if defined(NPY_PY3K)

#include <Python.h>

/*
 * PyInt -> PyLong
 */

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

/*
 * PyString -> PyBytes
 */

#define PyString_Type PyBytes_Type
#define PyString_Check PyBytes_Check

#define PyString_FromString PyBytes_FromString
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_FromFormat PyBytes_FromFormat
#define PyString_Concat PyBytes_Concat
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString
#define PyString_GET_SIZE PyBytes_GET_SIZE

#endif /* NPY_PY3K */

#endif /* _NPY_3KCOMPAT_H_ */

#ifndef _NPY_3KCOMPAT_H_
#define _NPY_3KCOMPAT_H_

#include <Python.h>
#include <stdio.h>

#include "npy_config.h"
#include "numpy/npy_common.h"

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
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_AsStringAndSize PyBytes_AsStringAndSize
#define PyString_FromFormat PyBytes_FromFormat
#define PyString_Concat PyBytes_Concat
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_Size PyBytes_Size

#define PyUString_Type PyUnicode_Type
#define PyUString_Check PyUnicode_Check
#define PyUStringObject PyUnicodeObject
#define PyUString_FromString PyUnicode_FromString
#define PyUString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyUString_FromFormat PyUnicode_FromFormat
#define PyUString_Concat PyUnicode_Concat2
#define PyUString_ConcatAndDel PyUnicode_ConcatAndDel
#define PyUString_GET_SIZE PyUnicode_GET_SIZE
#define PyUString_Size PyUnicode_Size
#define PyUString_InternFromString PyUnicode_InternFromString
#define PyUString_Format PyUnicode_Format

#else

#define PyBytes_Type PyString_Type
#define PyBytes_Check PyString_Check
#define PyBytesObject PyStringObject
#define PyBytes_FromString PyString_FromString
#define PyBytes_FromStringAndSize PyString_FromStringAndSize
#define PyBytes_AS_STRING PyString_AS_STRING
#define PyBytes_AsStringAndSize PyString_AsStringAndSize
#define PyBytes_FromFormat PyString_FromFormat
#define PyBytes_Concat PyString_Concat
#define PyBytes_ConcatAndDel PyString_ConcatAndDel
#define PyBytes_AsString PyString_AsString
#define PyBytes_GET_SIZE PyString_GET_SIZE
#define PyBytes_Size PyString_Size

#define PyUString_Type PyString_Type
#define PyUString_Check PyString_Check
#define PyUStringObject PyStringObject
#define PyUString_FromString PyString_FromString
#define PyUString_FromStringAndSize PyString_FromStringAndSize
#define PyUString_FromFormat PyString_FromFormat
#define PyUString_Concat PyString_Concat
#define PyUString_ConcatAndDel PyString_ConcatAndDel
#define PyUString_GET_SIZE PyString_GET_SIZE
#define PyUString_Size PyString_Size
#define PyUString_InternFromString PyString_InternFromString
#define PyUString_Format PyString_Format

#endif /* NPY_PY3K */


static NPY_INLINE void
PyUnicode_ConcatAndDel(PyObject **left, PyObject *right)
{
    PyObject *new;
    new = PyUnicode_Concat(*left, right);
    Py_DECREF(*left);
    Py_DECREF(right);
    *left = new;
}

static NPY_INLINE void
PyUnicode_Concat2(PyObject **left, PyObject *right)
{
    PyObject *new;
    new = PyUnicode_Concat(*left, right);
    Py_DECREF(*left);
    *left = new;
}


/*
 * Accessing items of ob_base
 */

#if (PY_VERSION_HEX < 0x02060000)
#define Py_TYPE(o)    (((PyObject*)(o))->ob_type)
#define Py_REFCNT(o)  (((PyObject*)(o))->ob_refcnt)
#define Py_SIZE(o)    (((PyVarObject*)(o))->ob_size)
#endif

/*
 * PyFile_AsFile
 */
#if defined(NPY_PY3K)
static NPY_INLINE FILE*
npy_PyFile_Dup(PyObject *file, char *mode)
{
    int fd, fd2;
    PyObject *ret, *os;
    /* Flush first to ensure things end up in the file in the correct order */
    ret = PyObject_CallMethod(file, "flush", "");
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return NULL;
    }
    os = PyImport_ImportModule("os");
    if (os == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(os, "dup", "i", fd);
    Py_DECREF(os);
    if (ret == NULL) {
        return NULL;
    }
    fd2 = PyNumber_AsSsize_t(ret, NULL);
    Py_DECREF(ret);
    return fdopen(fd2, mode);
}
#endif

static NPY_INLINE PyObject*
npy_PyFile_OpenFile(PyObject *filename, char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, "Os", filename, mode);
}

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

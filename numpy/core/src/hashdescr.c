#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

long PyArray_DescrHash(PyObject* odescr)
{
    PyArray_Descr *descr;
    int st;
    long hash;

    if (!PyArray_DescrCheck(odescr)) {
        PyErr_SetString(PyExc_ValueError,
                "PyArray_DescrHash argument must be a type descriptor");
        return -1;
    }
    descr = (PyArray_Descr*)odescr;

    hash = _Py_HashPointer(odescr);
    return hash;
}

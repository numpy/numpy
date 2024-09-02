#ifndef NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_
#define NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_

#include <Python.h>
#include "npy_pycompat.h"


static inline npy_bool
_is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* Basic number types */
        tp == &PyBool_Type ||
        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* Basic sequence types */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||

        /* other builtins */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        NPY_FALSE
    );
}


/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a numpy-specific one, so does not look
 * at builtin types. It does check base ndarray and numpy scalar types.
 *
 * It may make sense to just replace this with `PyObject_GetOptionalAttr`.
 */
static inline int
PyArray_LookupSpecial(
        PyObject *obj, PyObject *name_unicode, PyObject **res)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        *res = NULL;
        return 0;
    }

    return PyObject_GetOptionalAttr((PyObject *)tp, name_unicode, res);
}


/*
 * PyArray_LookupSpecial_OnInstance:
 *
 * Implements incorrect special method lookup rules, that break the python
 * convention, and looks on the instance, not the type.
 *
 * Kept for backwards compatibility. In future, we should deprecate this.
 */
static inline int
PyArray_LookupSpecial_OnInstance(
        PyObject *obj, PyObject *name_unicode, PyObject **res)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    /* Note: This check should likely be reduced on Python 3.13+ */
    if (_is_basic_python_type(tp)) {
        *res = NULL;
        return 0;
    }

    return PyObject_GetOptionalAttr(obj, name_unicode, res);
}

#endif  /* NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_ */

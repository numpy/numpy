#ifndef NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_
#define NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_

#include <Python.h>
#include "ufunc_object.h"

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
 * In future, could be made more like _Py_LookupSpecial
 */
static inline PyObject *
PyArray_LookupSpecial(PyObject *obj, PyObject *name_unicode)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }
    PyObject *res = PyObject_GetAttr((PyObject *)tp, name_unicode);

    if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    }

    return res;
}


/*
 * PyArray_LookupSpecial_OnInstance:
 *
 * Implements incorrect special method lookup rules, that break the python
 * convention, and looks on the instance, not the type.
 *
 * Kept for backwards compatibility. In future, we should deprecate this.
 */
static inline PyObject *
PyArray_LookupSpecial_OnInstance(PyObject *obj, PyObject *name_unicode)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }

    PyObject *res = PyObject_GetAttr(obj, name_unicode);

    if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    }

    return res;
}

#endif  /* NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_ */

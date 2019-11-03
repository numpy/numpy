#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "ufunc_override.h"

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns a new reference, the value of type(obj).__array_ufunc__ if it
 * exists and is different from that of ndarray, and NULL otherwise.
 */
NPY_NO_EXPORT PyObject *
PyUFuncOverride_GetNonDefaultArrayUfunc(PyObject *obj)
{
    static PyObject *ndarray_array_ufunc = NULL;
    PyObject *cls_array_ufunc;

    /* On first entry, cache ndarray's __array_ufunc__ */
    if (ndarray_array_ufunc == NULL) {
        ndarray_array_ufunc = PyObject_GetAttrString((PyObject *)&PyArray_Type,
                                                     "__array_ufunc__");
    }

    /* Fast return for ndarray */
    if (PyArray_CheckExact(obj)) {
        return NULL;
    }
    /*
     * Does the class define __array_ufunc__? (Note that LookupSpecial has fast
     * return for basic python types, so no need to worry about those here)
     */
    cls_array_ufunc = PyArray_LookupSpecial(obj, "__array_ufunc__");
    if (cls_array_ufunc == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
        }
        return NULL;
    }
    /* Ignore if the same as ndarray.__array_ufunc__ */
    if (cls_array_ufunc == ndarray_array_ufunc) {
        Py_DECREF(cls_array_ufunc);
        return NULL;
    }
    return cls_array_ufunc;
}

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns 1 if this is the case, 0 if not.
 */

NPY_NO_EXPORT int
PyUFunc_HasOverride(PyObject * obj)
{
    PyObject *method = PyUFuncOverride_GetNonDefaultArrayUfunc(obj);
    if (method) {
        Py_DECREF(method);
        return 1;
    }
    else {
        return 0;
    }
}

/*
 * Get possible out argument from kwds, and returns the number of outputs
 * contained within it: if a tuple, the number of elements in it, 1 otherwise.
 * The out argument itself is returned in out_kwd_obj, and the outputs
 * in the out_obj array (as borrowed references).
 *
 * Returns 0 if no outputs found, -1 if kwds is not a dict (with an error set).
 */
NPY_NO_EXPORT int
PyUFuncOverride_GetOutObjects(PyObject *kwds, PyObject **out_kwd_obj, PyObject ***out_objs)
{
    if (kwds == NULL) {
        Py_INCREF(Py_None);
        *out_kwd_obj = Py_None;
        return 0;
    }
    if (!PyDict_CheckExact(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                        "Internal Numpy error: call to PyUFuncOverride_GetOutObjects "
                        "with non-dict kwds");
        *out_kwd_obj = NULL;
        return -1;
    }
    /* borrowed reference */
    *out_kwd_obj = PyDict_GetItemString(kwds, "out");
    if (*out_kwd_obj == NULL) {
        Py_INCREF(Py_None);
        *out_kwd_obj = Py_None;
        return 0;
    }
    if (PyTuple_CheckExact(*out_kwd_obj)) {
        /*
         * The C-API recommends calling PySequence_Fast before any of the other
         * PySequence_Fast* functions. This is required for PyPy
         */
        PyObject *seq;
        seq = PySequence_Fast(*out_kwd_obj,
                              "Could not convert object to sequence");
        if (seq == NULL) {
            *out_kwd_obj = NULL;
            return -1;
        }
        *out_objs = PySequence_Fast_ITEMS(seq);
        *out_kwd_obj = seq;
        return PySequence_Fast_GET_SIZE(seq);
    }
    else {
        Py_INCREF(*out_kwd_obj);
        *out_objs = out_kwd_obj;
        return 1;
    }
}

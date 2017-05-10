#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"

#include "ufunc_override.h"

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns a new reference, the value of type(obj).__array_ufunc__
 *
 * If the __array_ufunc__ matches that of ndarray, or does not exist, return
 * NULL.
 *
 * Note that since this module is used with both multiarray and umath, we do
 * not have access to PyArray_Type and therewith neither to PyArray_CheckExact
 * nor to the default __array_ufunc__ method, so instead we import locally.
 * TODO: Can this really not be done more smartly?
 */
static PyObject *
get_non_default_array_ufunc(PyObject *obj)
{
    static PyObject *ndarray = NULL;
    static PyObject *ndarray_array_ufunc = NULL;
    PyObject *cls_array_ufunc;

    /* on first entry, import and cache ndarray and its __array_ufunc__ */
    if (ndarray == NULL) {
        npy_cache_import("numpy.core.multiarray", "ndarray", &ndarray);
        ndarray_array_ufunc = PyObject_GetAttrString(ndarray,
                                                     "__array_ufunc__");
    }

    /* Fast return for ndarray */
    if ((PyObject *)Py_TYPE(obj) == ndarray) {
        return NULL;
    }
    /* does the class define __array_ufunc__? */
    cls_array_ufunc = PyArray_LookupSpecial(obj, "__array_ufunc__");
    if (cls_array_ufunc == NULL) {
        return NULL;
    }
    /* is it different from ndarray.__array_ufunc__? */
    if (cls_array_ufunc != ndarray_array_ufunc) {
        return cls_array_ufunc;
    }
    Py_DECREF(cls_array_ufunc);
    return NULL;
}

/*
 * Check whether a set of input and output args have a non-default
 *  `__array_ufunc__` method. Return the number of overrides, setting
 * corresponding objects in PyObject array with_override and the corresponding
 * __array_ufunc__ methods in methods (both only if not NULL, and both using
 * new references).
 *
 * returns -1 on failure.
 */
NPY_NO_EXPORT int
PyUFunc_WithOverride(PyObject *args, PyObject *kwds,
                     PyObject **with_override, PyObject **methods)
{
    int i;

    int nargs;
    int nout_kwd = 0;
    int out_kwd_is_tuple = 0;
    int num_override_args = 0;

    PyObject *obj;
    PyObject *out_kwd_obj = NULL;
    /*
     * Check inputs
     */
    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_TypeError,
                        "Internal Numpy error: call to PyUFunc_HasOverride "
                        "with non-tuple");
        goto fail;
    }
    nargs = PyTuple_GET_SIZE(args);
    if (nargs > NPY_MAXARGS) {
        PyErr_SetString(PyExc_TypeError,
                        "Internal Numpy error: too many arguments in call "
                        "to PyUFunc_HasOverride");
        goto fail;
    }
    /* be sure to include possible 'out' keyword argument. */
    if (kwds && PyDict_CheckExact(kwds)) {
        out_kwd_obj = PyDict_GetItemString(kwds, "out");
        if (out_kwd_obj != NULL) {
            out_kwd_is_tuple = PyTuple_CheckExact(out_kwd_obj);
            if (out_kwd_is_tuple) {
                nout_kwd = PyTuple_GET_SIZE(out_kwd_obj);
            }
            else {
                nout_kwd = 1;
            }
        }
    }

    for (i = 0; i < nargs + nout_kwd; ++i) {
        PyObject *method;
        if (i < nargs) {
            obj = PyTuple_GET_ITEM(args, i);
        }
        else {
            if (out_kwd_is_tuple) {
                obj = PyTuple_GET_ITEM(out_kwd_obj, i - nargs);
            }
            else {
                obj = out_kwd_obj;
            }
        }
        /*
         * Now see if the object provides an __array_ufunc__. However, we should
         * ignore the base ndarray.__ufunc__, so we skip any ndarray as well as
         * any ndarray subclass instances that did not override __array_ufunc__.
         */
        method = get_non_default_array_ufunc(obj);
        if (method != NULL) {
            if (method == Py_None) {
                PyErr_Format(PyExc_TypeError,
                             "operand '%.200s' does not support ufuncs "
                             "(__array_ufunc__=None)",
                             obj->ob_type->tp_name);
                Py_DECREF(method);
                goto fail;
            }
            if (with_override != NULL) {
                Py_INCREF(obj);
                with_override[num_override_args] = obj;
            }
            if (methods != NULL) {
                methods[num_override_args] = method;
            }
            ++num_override_args;
        }
    }
    return num_override_args;

fail:
    if (methods != NULL) {
        for (i = 0; i < num_override_args; i++) {
            Py_XDECREF(methods[i]);
        }
    }
    return -1;
}

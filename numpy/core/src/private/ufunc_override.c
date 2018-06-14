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
NPY_NO_EXPORT PyObject *
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

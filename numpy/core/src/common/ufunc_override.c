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

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns 1 if this is the case, 0 if not.
 */

NPY_NO_EXPORT int
has_non_default_array_ufunc(PyObject * obj)
{
    PyObject *method = get_non_default_array_ufunc(obj);
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
 * in the out_obj array (all as borrowed references).
 *
 * Returns -1 if kwds is not a dict, 0 if no outputs found.
 */
static int
get_out_objects(PyObject *kwds, PyObject **out_kwd_obj, PyObject ***out_objs)
{
    if (kwds == NULL) {
        return 0;
    }
    if (!PyDict_CheckExact(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                        "Internal Numpy error: call to PyUFunc_WithOverride "
                        "with non-dict kwds");
        return -1;
    }
    /* borrowed reference */
    *out_kwd_obj = PyDict_GetItemString(kwds, "out");
    if (*out_kwd_obj == NULL) {
        return 0;
    }
    if (PyTuple_CheckExact(*out_kwd_obj)) {
        *out_objs = PySequence_Fast_ITEMS(*out_kwd_obj);
        return PySequence_Fast_GET_SIZE(*out_kwd_obj);
    }
    else {
        *out_objs = out_kwd_obj;
        return 1;
    }
}

/*
 * For each positional argument and each argument in a possible "out"
 * keyword, look for overrides of the standard ufunc behaviour, i.e.,
 * non-default __array_ufunc__ methods.
 *
 * Returns the number of overrides, setting corresponding objects
 * in PyObject array ``with_override`` and the corresponding
 * __array_ufunc__ methods in ``methods`` (both using new references).
 *
 * Only the first override for a given class is returned.
 *
 * returns -1 on failure.
 */
NPY_NO_EXPORT int
PyUFunc_WithOverride(PyObject *args, PyObject *kwds,
                     PyObject **with_override, PyObject **methods)
{
    int i;
    int num_override_args = 0;
    int narg, nout = 0;
    PyObject *out_kwd_obj;
    PyObject **arg_objs, **out_objs;

    narg = PyTuple_Size(args);
    if (narg < 0) {
        return -1;
    }
    arg_objs = PySequence_Fast_ITEMS(args);

    nout = get_out_objects(kwds, &out_kwd_obj, &out_objs);
    if (nout < 0) {
        return -1;
    }

    for (i = 0; i < narg + nout; ++i) {
        PyObject *obj;
        int j;
        int new_class = 1;

        if (i < narg) {
            obj = arg_objs[i];
        }
        else {
            obj = out_objs[i - narg];
        }
        /*
         * Have we seen this class before?  If so, ignore.
         */
        for (j = 0; j < num_override_args; j++) {
            new_class = (Py_TYPE(obj) != Py_TYPE(with_override[j]));
            if (!new_class) {
                break;
            }
        }
        if (new_class) {
            /*
             * Now see if the object provides an __array_ufunc__. However, we should
             * ignore the base ndarray.__ufunc__, so we skip any ndarray as well as
             * any ndarray subclass instances that did not override __array_ufunc__.
             */
            PyObject *method = get_non_default_array_ufunc(obj);
            if (method == NULL) {
                continue;
            }
            if (method == Py_None) {
                PyErr_Format(PyExc_TypeError,
                             "operand '%.200s' does not support ufuncs "
                             "(__array_ufunc__=None)",
                             obj->ob_type->tp_name);
                Py_DECREF(method);
                goto fail;
            }
            Py_INCREF(obj);
            with_override[num_override_args] = obj;
            methods[num_override_args] = method;
            ++num_override_args;
        }
    }
    return num_override_args;

fail:
    for (i = 0; i < num_override_args; i++) {
        Py_DECREF(with_override[i]);
        Py_DECREF(methods[i]);
    }
    return -1;
}

/*
 * Check whether any of a set of input and output args have a non-default
 * __array_ufunc__ method. Return 1 if so, 0 if not.
 *
 * This function primarily exists to help ndarray.__array_ufunc__ determine
 * whether it can support a ufunc (which is the case only if none of the
 * operands have an override).  Thus, unlike in PyUFunc_CheckOverride, the
 * actual overrides are not needed and one can stop looking once one is found.
 *
 * TODO: move this function and has_non_default_array_ufunc closer to ndarray.
 */
NPY_NO_EXPORT int
PyUFunc_HasOverride(PyObject *args, PyObject *kwds)
{
    int i;
    int nin, nout;
    PyObject *out_kwd_obj;
    PyObject **in_objs, **out_objs;

    /* check inputs */
    nin = PyTuple_Size(args);
    if (nin < 0) {
        return -1;
    }
    in_objs = PySequence_Fast_ITEMS(args);
    for (i = 0; i < nin; ++i) {
        if (has_non_default_array_ufunc(in_objs[i])) {
            return 1;
        }
    }
    /* check outputs, if any */
    nout = get_out_objects(kwds, &out_kwd_obj, &out_objs);
    if (nout < 0) {
        return -1;
    }
    for (i = 0; i < nout; i++) {
        if (has_non_default_array_ufunc(out_objs[i])) {
            return 1;
        }
    }
    return 0;
}

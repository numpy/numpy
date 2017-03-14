#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "npy_pycompat.h"
#include "numpy/ufuncobject.h"
#include "get_attr_string.h"

#include "ufunc_override.h"

static void
normalize___call___args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds,
                    int nin)
{
    /* ufunc.__call__(*args, **kwds) */
    int i;
    int not_all_none;
    int nargs = PyTuple_GET_SIZE(args);
    PyObject *obj = PyDict_GetItemString(*normal_kwds, "sig");

    /* ufuncs accept 'sig' or 'signature' normalize to 'signature' */
    if (obj != NULL) {
        Py_INCREF(obj);
        PyDict_SetItemString(*normal_kwds, "signature", obj);
        PyDict_DelItemString(*normal_kwds, "sig");
    }

    *normal_args = PyTuple_GetSlice(args, 0, nin);

    /* If we have more args than nin, they must be the output variables.*/
    if (nargs > nin) {
        for (i=nin; i < nargs; i++) {
            not_all_none = (PyTuple_GET_ITEM(args, i) != Py_None);
            if (not_all_none) {
                break;
            }
        }
        if (not_all_none) {
            obj = PyTuple_GetSlice(args, nin, nargs);
            PyDict_SetItemString(*normal_kwds, "out", obj);
            Py_DECREF(obj);
        }
    }
}

static void
normalize_reduce_args(PyUFuncObject *ufunc, PyObject *args,
                  PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.reduce(a[, axis, dtype, out, keepdims]) */
    int nargs = PyTuple_GET_SIZE(args);
    int i;
    PyObject *obj;

    *normal_args = PyTuple_GetSlice(args, 0, 1);
    for (i = 1; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (obj == Py_None) {
            continue;
        }
        if (i == 1) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 2) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else if (i == 3) {
            /* out */
            obj = PyTuple_GetSlice(args, 3, 4);
            PyDict_SetItemString(*normal_kwds, "out", obj);
            Py_DECREF(obj);
        }
        else {
            /* keepdims */
            PyDict_SetItemString(*normal_kwds, "keepdims", obj);
        }
    }
    return;
}

static void
normalize_accumulate_args(PyUFuncObject *ufunc, PyObject *args,
                      PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.accumulate(a[, axis, dtype, out]) */
    int nargs = PyTuple_GET_SIZE(args);
    int i;
    PyObject *obj;

    *normal_args = PyTuple_GetSlice(args, 0, 1);
    for (i = 1; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (obj == Py_None) {
            continue;
        }
        if (i == 1) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 2) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else {
            /* out */
            obj = PyTuple_GetSlice(args, 3, 4);
            PyDict_SetItemString(*normal_kwds, "out", obj);
            Py_DECREF(obj);
        }
    }
    return;
}

static void
normalize_reduceat_args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.reduceat(a, indicies[, axis, dtype, out]) */
    int i;
    int nargs = PyTuple_GET_SIZE(args);
    PyObject *obj;

    /* a and indicies */
    *normal_args = PyTuple_GetSlice(args, 0, 2);

    for (i = 2; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (obj == Py_None) {
            continue;
        }
        if (i == 2) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 3) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else {
            /* out */
            obj = PyTuple_GetSlice(args, 4, 5);
            PyDict_SetItemString(*normal_kwds, "out", obj);
            Py_DECREF(obj);
        }
    }
    return;
}

static void
normalize_outer_args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.outer(A, B)
     * This has no kwds so we don't need to do any kwd stuff.
     */
    *normal_args = PyTuple_GetSlice(args, 0, 2);
    return;
}

static void
normalize_at_args(PyUFuncObject *ufunc, PyObject *args,
                  PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.at(a, indices[, b]) */
    int nargs = PyTuple_GET_SIZE(args);

    *normal_args = PyTuple_GetSlice(args, 0, nargs);
    return;
}

/*
 * Check a set of args for the `__array_ufunc__` method.  If more than one of
 * the input arguments implements `__array_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * (non-None) routine returning something other than `NotImplemented`
 * determines the result. If all of the `__array_ufunc__` operations return
 * `NotImplemented` (or are None), a `TypeError` is raised.
 *
 * Returns 0 on success and 1 on exception. On success, *result contains the
 * result of the operation, if any. If *result is NULL, there is no override.
 */
NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
                      PyObject *args, PyObject *kwds,
                      PyObject **result,
                      int nin)
{
    int i;
    int j;

    int nargs;
    int nout_kwd = 0;
    int out_kwd_is_tuple = 0;
    int noa = 0; /* Number of overriding args.*/

    PyObject *tmp;
    PyObject *obj;
    PyObject *out_kwd_obj = NULL;
    PyObject *other_obj;

    PyObject *method_name = NULL;
    PyObject *normal_args = NULL; /* normal_* holds normalized arguments. */
    PyObject *normal_kwds = NULL;

    PyObject *with_override[NPY_MAXARGS];
    Py_ssize_t len;
    PyObject *override_args;

    /*
     * Check inputs
     */
    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_ValueError,
                        "Internal Numpy error: call to PyUFunc_CheckOverride "
                        "with non-tuple");
        goto fail;
    }
    nargs = PyTuple_GET_SIZE(args);
    if (nargs > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                        "Internal Numpy error: too many arguments in call "
                        "to PyUFunc_CheckOverride");
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
        if (i < nargs) {
            obj = PyTuple_GET_ITEM(args, i);
        }
        else {
            if (out_kwd_is_tuple) {
                obj = PyTuple_GET_ITEM(out_kwd_obj, i-nargs);
            }
            else {
                obj = out_kwd_obj;
            }
        }
        tmp = PyArray_GetAttrString_SuppressException(obj, "__array_ufunc__");
        if (tmp) {
            Py_DECREF(tmp);
            with_override[noa] = obj;
            ++noa;
        }
    }

    /* No overrides, bail out.*/
    if (noa == 0) {
        *result = NULL;
        return 0;
    }

    method_name = PyUString_FromString(method);
    if (method_name == NULL) {
        goto fail;
    }

    /*
     * Normalize ufunc arguments.
     */

    /* Build new kwds */
    if (kwds && PyDict_CheckExact(kwds)) {
        PyObject *out;

        /* ensure out is always a tuple */
        normal_kwds = PyDict_Copy(kwds);
        out = PyDict_GetItemString(normal_kwds, "out");
        if (out != NULL) {
            if (PyTuple_Check(out)) {
                int all_none;
                int i;

                for (i = 0; i < PyTuple_GET_SIZE(out); i++) {
                    all_none = (PyTuple_GET_ITEM(out, i) == Py_None);
                    if (!all_none) {
                        break;
                    }
                }
                if (all_none) {
                    PyDict_DelItemString(normal_kwds, "out");
                }
            }
            else if (out != Py_None) {
                /* not already a tuple and not None */
                PyObject *out_tuple = PyTuple_New(1);

                if (out_tuple == NULL) {
                    goto fail;
                }
                /* out was borrowed ref; make it permanent */
                Py_INCREF(out);
                /* steals reference */
                PyTuple_SET_ITEM(out_tuple, 0, out);
                PyDict_SetItemString(normal_kwds, "out", out_tuple);
                Py_DECREF(out_tuple);
            }
        }
    }
    else {
        normal_kwds = PyDict_New();
    }
    if (normal_kwds == NULL) {
        goto fail;
    }

    /* decide what to do based on the method. */

    /* ufunc.__call__ */
    if (strcmp(method, "__call__") == 0) {
        normalize___call___args(ufunc, args, &normal_args, &normal_kwds, nin);
    }

    /* ufunc.reduce */
    else if (strcmp(method, "reduce") == 0) {
        normalize_reduce_args(ufunc, args, &normal_args, &normal_kwds);
    }

    /* ufunc.accumulate */
    else if (strcmp(method, "accumulate") == 0) {
        normalize_accumulate_args(ufunc, args, &normal_args, &normal_kwds);
    }

    /* ufunc.reduceat */
    else if (strcmp(method, "reduceat") == 0) {
        normalize_reduceat_args(ufunc, args, &normal_args, &normal_kwds);
    }

    /* ufunc.outer */
    else if (strcmp(method, "outer") == 0) {
        normalize_outer_args(ufunc, args, &normal_args, &normal_kwds);
    }

    /* ufunc.at */
    else if (strcmp(method, "at") == 0) {
        normalize_at_args(ufunc, args, &normal_args, &normal_kwds);
    }

    if (normal_args == NULL) {
        goto fail;
    }

    len = PyTuple_GET_SIZE(normal_args);
    override_args = PyTuple_New(len + 2);
    if (override_args == NULL) {
        goto fail;
    }

    Py_INCREF(ufunc);
    /* PyTuple_SET_ITEM steals reference */
    PyTuple_SET_ITEM(override_args, 0, ufunc);
    Py_INCREF(method_name);
    PyTuple_SET_ITEM(override_args, 1, method_name);
    for (i = 0; i < len; i++) {
        PyObject *item = PyTuple_GET_ITEM(normal_args, i);

        Py_INCREF(item);
        PyTuple_SET_ITEM(override_args, i + 2, item);
    }
    Py_DECREF(normal_args);

    /* Call __array_ufunc__ functions in correct order */
    while (1) {
        PyObject *array_ufunc;
        PyObject *override_obj;

        override_obj = NULL;
        *result = NULL;

        /* Choose an overriding argument */
        for (i = 0; i < noa; i++) {
            obj = with_override[i];
            if (obj == NULL) {
                continue;
            }

            /* Get the first instance of an overriding arg.*/
            override_obj = obj;

            /* Check for sub-types to the right of obj. */
            for (j = i + 1; j < noa; j++) {
                other_obj = with_override[j];
                if (PyObject_Type(other_obj) != PyObject_Type(obj) &&
                    PyObject_IsInstance(other_obj,
                                        PyObject_Type(override_obj))) {
                    override_obj = NULL;
                    break;
                }
            }

            /* override_obj had no subtypes to the right. */
            if (override_obj) {
                /* We won't call this one again */
                with_override[i] = NULL;
                break;
            }
        }

        /* Check if there is a method left to call */
        if (!override_obj) {
            /* No acceptable override found. */
            PyErr_SetString(PyExc_TypeError,
                            "__array_ufunc__ not implemented for this type.");
            goto fail;
        }

        /* Access the override */
        array_ufunc = PyObject_GetAttrString(override_obj,
                                             "__array_ufunc__");
        if (array_ufunc == NULL) {
            goto fail;
        }

        /* If None, try next one (i.e., as if it returned NotImplemented) */
        if (array_ufunc == Py_None) {
            Py_DECREF(array_ufunc);
            continue;
        }

        *result = PyObject_Call(array_ufunc, override_args, normal_kwds);
        Py_DECREF(array_ufunc);

        if (*result == NULL) {
            /* Exception occurred */
            goto fail;
        }
        else if (*result == Py_NotImplemented) {
            /* Try the next one */
            Py_DECREF(*result);
            continue;
        }
        else {
            /* Good result. */
            break;
        }
    }

    /* Override found, return it. */
    Py_XDECREF(method_name);
    Py_XDECREF(normal_kwds);
    Py_DECREF(override_args);
    return 0;

fail:
    Py_XDECREF(method_name);
    Py_XDECREF(normal_kwds);
    Py_XDECREF(override_args);
    return 1;
}

#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H
#include <npy_config.h>
#include "numpy/arrayobject.h"
#include "common.h"
#include <string.h>
#include "numpy/ufuncobject.h"

static void
normalize___call___args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds,
                    int nin)
{
    /* ufunc.__call__(*args, **kwds) */
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
        if ((nargs - nin) == 1) {
            obj = PyTuple_GET_ITEM(args, nargs - 1);
            PyDict_SetItemString(*normal_kwds, "out", obj);
        }
        else {
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

    for (i = 0; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 0) {
            *normal_args = PyTuple_GetSlice(args, 0, 1);
        }
        else if (i == 1) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 2) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else if (i == 3) {
            /* out */
            PyDict_SetItemString(*normal_kwds, "out", obj);
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

    for (i = 0; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 0) {
            *normal_args = PyTuple_GetSlice(args, 0, 1);
        }
        else if (i == 1) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 2) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else {
            /* out */
            PyDict_SetItemString(*normal_kwds, "out", obj);
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

    for (i = 0; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 0) {
            /* a and indicies */
            *normal_args = PyTuple_GetSlice(args, 0, 2);
        }
        else if (i == 1) {
            /* Handled above, when i == 0. */
            continue;
        }
        else if (i == 2) {
            /* axis */
            PyDict_SetItemString(*normal_kwds, "axis", obj);
        }
        else if (i == 3) {
            /* dtype */
            PyDict_SetItemString(*normal_kwds, "dtype", obj);
        }
        else {
            /* out */
            PyDict_SetItemString(*normal_kwds, "out", obj);
        }
    }
    return;
}

static void
normalize_outer_args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.outer(A, B)
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
 * Check a set of args for the `__numpy_ufunc__` method.  If more than one of
 * the input arguments implements `__numpy_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * routine returning something other than `NotImplemented` determines the
 * result. If all of the `__numpy_ufunc__` operations returns `NotImplemented`,
 * a `TypeError` is raised.
 *
 * Returns 0 on success and 1 on exception. On success, *result contains the
 * result of the operation, if any. If *result is NULL, there is no override.
 */
static int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
                      PyObject *args, PyObject *kwds,
                      PyObject **result,
                      int nin)
{
    int i;
    int override_pos; /* Position of override in args.*/
    int j;

    int nargs;
    int nout_kwd = 0;
    int out_kwd_is_tuple = 0;
    int noa = 0; /* Number of overriding args.*/

    PyObject *obj;
    PyObject *out_kwd_obj = NULL;
    PyObject *other_obj;

    PyObject *method_name = NULL;
    PyObject *normal_args = NULL; /* normal_* holds normalized arguments. */
    PyObject *normal_kwds = NULL;

    PyObject *with_override[NPY_MAXARGS];

    /* Pos of each override in args */
    int with_override_pos[NPY_MAXARGS];

    /* 2016-01-29: Disable for now in master -- can re-enable once details are
     * sorted out. All commented bits are tagged NUMPY_UFUNC_DISABLED. -njs
     */
    result = NULL;
    return 0;

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
    if ((kwds)&& (PyDict_CheckExact(kwds))) {
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
        /*
         * TODO: could use PyArray_GetAttrString_SuppressException if it
         * weren't private to multiarray.so
         */
        if (PyArray_CheckExact(obj) || PyArray_IsScalar(obj, Generic) ||
            _is_basic_python_type(obj)) {
            continue;
        }
        if (PyObject_HasAttrString(obj, "__numpy_ufunc__")) {
            with_override[noa] = obj;
            with_override_pos[noa] = i;
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
        normal_kwds = PyDict_Copy(kwds);
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

    /*
     * Call __numpy_ufunc__ functions in correct order
     */
    while (1) {
        PyObject *numpy_ufunc;
        PyObject *override_args;
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
            override_pos = with_override_pos[i];
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
                with_override[i] = NULL; /* We won't call this one again */
                break;
            }
        }

        /* Check if there is a method left to call */
        if (!override_obj) {
            /* No acceptable override found. */
            PyErr_SetString(PyExc_TypeError,
                            "__numpy_ufunc__ not implemented for this type.");
            goto fail;
        }

        /* Call the override */
        numpy_ufunc = PyObject_GetAttrString(override_obj,
                                             "__numpy_ufunc__");
        if (numpy_ufunc == NULL) {
            goto fail;
        }

        override_args = Py_BuildValue("OOiO", ufunc, method_name,
                                      override_pos, normal_args);
        if (override_args == NULL) {
            Py_DECREF(numpy_ufunc);
            goto fail;
        }

        *result = PyObject_Call(numpy_ufunc, override_args, normal_kwds);

        Py_DECREF(numpy_ufunc);
        Py_DECREF(override_args);

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
    Py_XDECREF(normal_args);
    Py_XDECREF(normal_kwds);
    return 0;

fail:
    Py_XDECREF(method_name);
    Py_XDECREF(normal_args);
    Py_XDECREF(normal_kwds);
    return 1;
}
#endif

/*
 * Definitions for dealing with array-wrap or array-prepare.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/npy_math.h"
#include "get_attr_string.h"

#include "arraywrap.h"
#include "npy_static_data.h"

/*
 * Find the array wrap or array prepare method that applies to the inputs.
 * outputs should NOT be passed, as they are considered individually while
 * applying the wrapping.
 *
 * @param nin number of inputs
 * @param inputs Original input objects
 * @param out_wrap Set to the python callable or None (on success).
 * @param out_wrap_type Set to the type belonging to the wrapper.
 */
NPY_NO_EXPORT int
npy_find_array_wrap(
        int nin, PyObject *const *inputs,
        PyObject **out_wrap, PyObject **out_wrap_type)
{
    PyObject *wrap = NULL;
    PyObject *wrap_type = NULL;

    double priority = -NPY_INFINITY;

    /*
     * Iterate through all inputs taking the first one with an __array_wrap__
     * and replace it if a later one has a higher priority.
     * (Currently even priority=-inf can be picked if it is the only argument.)
     */
    for (int i = 0; i < nin; i++) {
        PyObject *obj = inputs[i];
        if (PyArray_CheckExact(obj)) {
            if (priority < NPY_PRIORITY) {
                Py_XSETREF(wrap, Py_NewRef(Py_None));
                priority = NPY_PRIORITY;
            }
        }
        else if (PyArray_IsAnyScalar(obj)) {
            if (priority < NPY_SCALAR_PRIORITY) {
                Py_XSETREF(wrap, Py_NewRef(Py_None));
                priority = NPY_SCALAR_PRIORITY;
            }
        }
        else {
            PyObject *new_wrap;
            if (PyArray_LookupSpecial_OnInstance(
                    obj, npy_interned_str.array_wrap, &new_wrap) < 0) {
                goto fail;
            }
            else if (new_wrap == NULL) {
                continue;
            }
            double curr_priority = PyArray_GetPriority(obj, NPY_PRIORITY);
            if (wrap == NULL || priority < curr_priority
                    /* Prefer subclasses `__array_wrap__`: */
                    || (curr_priority == NPY_PRIORITY && wrap == Py_None)) {
                Py_XSETREF(wrap, new_wrap);
                Py_XSETREF(wrap_type, Py_NewRef(Py_TYPE(obj)));
                priority = curr_priority;
            }
            else {
                Py_DECREF(new_wrap);
            }
        }
    }

    if (wrap == NULL) {
        wrap = Py_NewRef(Py_None);
    }
    if (wrap_type == NULL) {
        wrap_type = Py_NewRef(&PyArray_Type);
    }

    *out_wrap = wrap;
    *out_wrap_type = wrap_type;

    return 0;

  fail:
    Py_XDECREF(wrap);
    Py_XDECREF(wrap_type);
    return -1;
}


/* Get the arg tuple to pass in the context argument to __array_wrap__.
 *
 * Output arguments are only passed if at least one is non-None.
 */
static PyObject *
_get_wrap_prepare_args(NpyUFuncContext *context) {
    if (context->out == NULL) {
        Py_INCREF(context->in);
        return context->in;
    }
    else {
        return PySequence_Concat(context->in, context->out);
    }
}


/*
 * Apply the array wrapping to a result array.
 *
 * @param obj The object (should be an array) to wrap.
 * @param original_out NULL/None (both valid) or an object who's wrapping
 *        method is always used/preferred.  The naming comes because for an
 *        `out=` argument we always trigger its own wrapping method.
 * @param wrap The array wrap function to call
 * @param wrap_type The type belonging to the wrap function, when it matches
 *        wrapping may be short-cut.
 * @param context ufunc context or NULL, only used by normal ufunc calls.
 * @param return_scalar Whether to prefer a scalar return.  Ignored when
 *        `original_out` is passed, as an `out` argument is never scalar.
 * @param force_wrap If True, we call the wrap (for subclasses) always,
 *        because the ufunc would have mutated the content.
 */
NPY_NO_EXPORT PyObject *
npy_apply_wrap(
        PyObject *obj, PyObject *original_out,
        PyObject *wrap, PyObject *wrap_type,
        NpyUFuncContext *context, npy_bool return_scalar, npy_bool force_wrap)
{
    PyObject *res = NULL;
    PyObject *new_wrap = NULL;
    PyArrayObject *arr = NULL;
    PyObject *err_type, *err_value, *traceback;

    /* If provided, we prefer the actual out objects wrap: */
    if (original_out != NULL && original_out != Py_None) {
        /*
         * If an original output object was passed, wrapping shouldn't
         * change it.  In particular, it doesn't make sense to convert to
         * scalar.  So replace the passed in wrap and wrap_type.
         */
        return_scalar = NPY_FALSE;

        if (PyArray_CheckExact(original_out)) {
            /* Replace passed wrap/wrap_type (borrowed refs) with default. */
            wrap = Py_None;
            wrap_type = (PyObject *)&PyArray_Type;
        }
        else {
            /* Replace passed wrap/wrap_type (borrowed refs) with new_wrap/type. */
            if (PyArray_LookupSpecial_OnInstance(
                    original_out, npy_interned_str.array_wrap, &new_wrap) < 0) {
                return NULL;
            }
            else if (new_wrap != NULL) {
                wrap = new_wrap;
                wrap_type = (PyObject *)Py_TYPE(original_out);
            }
        }
    }
    /*
     * If the result is the same type as the wrapping (and there is no
     * `original_out`, when we should be wrapping `self` probably)
     * we can skip wrapping, unless we need the scalar return.
     */
    if (!return_scalar && !force_wrap
            && (PyObject *)Py_TYPE(obj) == wrap_type) {
        Py_XDECREF(new_wrap);
        Py_INCREF(obj);
        return obj;
    }

    if (wrap == Py_None) {
        Py_XDECREF(new_wrap);
        Py_INCREF(obj);
        if (return_scalar) {
            /*
             * Use PyArray_Return to convert to scalar when necessary
             * (PyArray_Return actually checks for non-arrays).
             */
            return PyArray_Return((PyArrayObject *)obj);
        }
        else {
            return obj;
        }
    }

    /*
     * We have to call array-wrap.  In some branches, input might be non-array.
     * (We should try to phase all of these out, though!)
     */
    PyObject *py_context = NULL;
    if (context == NULL) {
        Py_INCREF(Py_None);
        py_context = Py_None;
    }
    else {
        /* Call the method with appropriate context */
        PyObject *args_tup = _get_wrap_prepare_args(context);
        if (args_tup == NULL) {
            goto finish;
        }
        py_context = Py_BuildValue("OOi",
                context->ufunc, args_tup, context->out_i);
        Py_DECREF(args_tup);
        if (py_context == NULL) {
            goto finish;
        }
    }

    if (PyArray_Check(obj)) {
        Py_INCREF(obj);
        arr = (PyArrayObject *)obj;
    }
    else {
        /*
         * TODO: Ideally, we never end up in this branch!  But when we use
         *       this from Python and NumPy's (current) habit of converting
         *       0-d arrays to scalars means that it would be odd to convert
         *       back to an array in python when we may not need to wrap.
         */
        arr = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
        if (arr == NULL) {
            goto finish;
        }
    }

    res = PyObject_CallFunctionObjArgs(
            wrap, arr, py_context,
            (return_scalar && PyArray_NDIM(arr) == 0) ? Py_True : Py_False,
            NULL);
    if (res != NULL) {
        goto finish;
    }
    else if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        goto finish;
    }

    /*
     * Retry without passing return_scalar.  If that succeeds give a
     * Deprecation warning.
     * When context is None, there is no reason to try this, though.
     */
    if (py_context != Py_None) {
        PyErr_Fetch(&err_type, &err_value, &traceback);
        res = PyObject_CallFunctionObjArgs(wrap, arr, py_context, NULL);
        if (res != NULL) {
            goto deprecation_warning;
        }
        Py_DECREF(err_type);
        Py_XDECREF(err_value);
        Py_XDECREF(traceback);
        if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
            goto finish;
        }
    }

    /*
     * Retry without passing context and return_scalar parameters.
     * If that succeeds, we give a DeprecationWarning.
     */
    PyErr_Fetch(&err_type, &err_value, &traceback);
    res = PyObject_CallFunctionObjArgs(wrap, arr, NULL);
    if (res == NULL) {
        Py_DECREF(err_type);
        Py_XDECREF(err_value);
        Py_XDECREF(traceback);
        goto finish;
    }

  deprecation_warning:
    /* If we reach here, the original error is still stored. */
    /* Deprecated 2024-01-17, NumPy 2.0 */
    if (DEPRECATE(
            "__array_wrap__ must accept context and return_scalar arguments "
            "(positionally) in the future. (Deprecated NumPy 2.0)") < 0) {
        npy_PyErr_ChainExceptionsCause(err_type, err_value, traceback);
        Py_CLEAR(res);
    }
    else {
        Py_DECREF(err_type);
        Py_XDECREF(err_value);
        Py_XDECREF(traceback);
    }

  finish:
    Py_XDECREF(py_context);
    Py_XDECREF(arr);
    Py_XDECREF(new_wrap);
    return res;
}


/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 * `towrap` should be a base-class ndarray.
 */
NPY_NO_EXPORT PyObject *
npy_apply_wrap_simple(PyArrayObject *arr_of_subclass, PyArrayObject *towrap)
{
    /*
     * Same as apply-wrap, but when there is only a single other array we
     * can  `original_out` and not worry about passing a useful wrap.
     */
    return npy_apply_wrap(
            (PyObject *)towrap, (PyObject *)arr_of_subclass, Py_None, NULL,
            NULL, NPY_FALSE, NPY_TRUE);
}

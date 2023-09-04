#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#include "npy_pycompat.h"
#include "numpy/ufuncobject.h"
#include "npy_import.h"

#include "override.h"
#include "ufunc_override.h"


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
 * Returns -1 on failure.
 */
static int
get_array_ufunc_overrides(PyObject *in_args, PyObject *out_args, PyObject *wheremask_obj,
                          PyObject **with_override, PyObject **methods)
{
    int i;
    int num_override_args = 0;
    int narg, nout, nwhere;

    narg = (int)PyTuple_GET_SIZE(in_args);
    /* It is valid for out_args to be NULL: */
    nout = (out_args != NULL) ? (int)PyTuple_GET_SIZE(out_args) : 0;
    nwhere = (wheremask_obj != NULL) ? 1: 0;

    for (i = 0; i < narg + nout + nwhere; ++i) {
        PyObject *obj;
        int j;
        int new_class = 1;

        if (i < narg) {
            obj = PyTuple_GET_ITEM(in_args, i);
        }
        else if (i < narg + nout){
            obj = PyTuple_GET_ITEM(out_args, i - narg);
        }
        else {
            obj = wheremask_obj;
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
            PyObject *method = PyUFuncOverride_GetNonDefaultArrayUfunc(obj);
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
 * Build a dictionary from the keyword arguments, but replace out with the
 * normalized version (and always pass it even if it was passed by position).
 */
static int
initialize_normal_kwds(PyObject *out_args,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject *normal_kwds)
{
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); i++) {
            if (PyDict_SetItem(normal_kwds,
                    PyTuple_GET_ITEM(kwnames, i), args[i + len_args]) < 0) {
                return -1;
            }
        }
    }
    static PyObject *out_str = NULL;
    if (out_str == NULL) {
        out_str = PyUnicode_InternFromString("out");
        if (out_str == NULL) {
            return -1;
        }
    }

    if (out_args != NULL) {
        /* Replace `out` argument with the normalized version */
        int res = PyDict_SetItem(normal_kwds, out_str, out_args);
        if (res < 0) {
            return -1;
        }
    }
    else {
        /* Ensure that `out` is not present. */
        int res = PyDict_Contains(normal_kwds, out_str);
        if (res < 0) {
            return -1;
        }
        if (res) {
            return PyDict_DelItem(normal_kwds, out_str);
        }
    }
    return 0;
}

/*
 * ufunc() and ufunc.outer() accept 'sig' or 'signature'.  We guarantee
 * that it is passed as 'signature' by renaming 'sig' if present.
 * Note that we have already validated that only one of them was passed
 * before checking for overrides.
 */
static int
normalize_signature_keyword(PyObject *normal_kwds)
{
    /* If the keywords include `sig` rename to `signature`. */
    PyObject* obj = _PyDict_GetItemStringWithError(normal_kwds, "sig");
    if (obj == NULL && PyErr_Occurred()) {
        return -1;
    }
    if (obj != NULL) {
        /*
         * No INCREF or DECREF needed: got a borrowed reference above,
         * and, unlike e.g. PyList_SetItem, PyDict_SetItem INCREF's it.
         */
        if (PyDict_SetItemString(normal_kwds, "signature", obj) < 0) {
            return -1;
        }
        if (PyDict_DelItemString(normal_kwds, "sig") < 0) {
            return -1;
        }
    }
    return 0;
}


static int
copy_positional_args_to_kwargs(const char **keywords,
        PyObject *const *args, Py_ssize_t len_args,
        PyObject *normal_kwds)
{
    for (Py_ssize_t i = 0; i < len_args; i++) {
        if (keywords[i] == NULL) {
            /* keyword argument is either input or output and not set here */
            continue;
        }
        if (NPY_UNLIKELY(i == 5)) {
            /*
             * This is only relevant for reduce, which is the only one with
             * 5 keyword arguments.
             */
            static PyObject *NoValue = NULL;
            assert(strcmp(keywords[i], "initial") == 0);
            npy_cache_import("numpy", "_NoValue", &NoValue);
            if (args[i] == NoValue) {
                continue;
            }
        }

        int res = PyDict_SetItemString(normal_kwds, keywords[i], args[i]);
        if (res < 0) {
            return -1;
        }
    }
    return 0;
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
        PyObject *in_args, PyObject *out_args, PyObject *wheremask_obj,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result)
{
    int status;

    int num_override_args;
    PyObject *with_override[NPY_MAXARGS];
    PyObject *array_ufunc_methods[NPY_MAXARGS];

    PyObject *method_name = NULL;
    PyObject *normal_kwds = NULL;

    PyObject *override_args = NULL;

    /*
     * Check inputs for overrides
     */
    num_override_args = get_array_ufunc_overrides(
           in_args, out_args, wheremask_obj, with_override, array_ufunc_methods);
    if (num_override_args == -1) {
        goto fail;
    }
    /* No overrides, bail out.*/
    if (num_override_args == 0) {
        *result = NULL;
        return 0;
    }

    /*
     * Normalize ufunc arguments, note that any input and output arguments
     * have already been stored in `in_args` and `out_args`.
     */
    normal_kwds = PyDict_New();
    if (normal_kwds == NULL) {
        goto fail;
    }
    if (initialize_normal_kwds(out_args,
            args, len_args, kwnames, normal_kwds) < 0) {
        goto fail;
    }

    /*
     * Reduce-like methods can pass keyword arguments also by position,
     * in which case the additional positional arguments have to be copied
     * into the keyword argument dictionary. The `__call__` and `__outer__`
     * method have to normalize sig and signature.
     */

    /* ufunc.__call__ */
    if (strcmp(method, "__call__") == 0) {
        status = normalize_signature_keyword(normal_kwds);
    }
    /* ufunc.reduce */
    else if (strcmp(method, "reduce") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL, "keepdims",
                "initial", "where"};
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.accumulate */
    else if (strcmp(method, "accumulate") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL};
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.reduceat */
    else if (strcmp(method, "reduceat") == 0) {
        static const char *keywords[] = {
                NULL, NULL, "axis", "dtype", NULL};
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.outer (identical to call) */
    else if (strcmp(method, "outer") == 0) {
        status = normalize_signature_keyword(normal_kwds);
    }
    /* ufunc.at */
    else if (strcmp(method, "at") == 0) {
        status = 0;
    }
    /* unknown method */
    else {
        PyErr_Format(PyExc_TypeError,
                     "Internal Numpy error: unknown ufunc method '%s' in call "
                     "to PyUFunc_CheckOverride", method);
        status = -1;
    }
    if (status != 0) {
        goto fail;
    }

    method_name = PyUnicode_FromString(method);
    if (method_name == NULL) {
        goto fail;
    }

    int len = (int)PyTuple_GET_SIZE(in_args);

    /* Call __array_ufunc__ functions in correct order */
    while (1) {
        PyObject *override_obj = NULL;
        PyObject *override_array_ufunc = NULL;

        *result = NULL;

        /* Choose an overriding argument */
        for (int i = 0; i < num_override_args; i++) {
            override_obj = with_override[i];
            if (override_obj == NULL) {
                continue;
            }

            /* Check for sub-types to the right of obj. */
            for (int j = i + 1; j < num_override_args; j++) {
                PyObject *other_obj = with_override[j];
                if (other_obj != NULL &&
                    Py_TYPE(other_obj) != Py_TYPE(override_obj) &&
                    PyObject_IsInstance(other_obj,
                                        (PyObject *)Py_TYPE(override_obj))) {
                    override_obj = NULL;
                    break;
                }
            }

            /* override_obj had no subtypes to the right. */
            if (override_obj) {
                override_array_ufunc = array_ufunc_methods[i];
                /* We won't call this one again (references decref'd below) */
                with_override[i] = NULL;
                array_ufunc_methods[i] = NULL;
                break;
            }
        }
        /*
         * Set override arguments for each call since the tuple must
         * not be mutated after use in PyPy
         * We increase all references since SET_ITEM steals
         * them and they will be DECREF'd when the tuple is deleted.
         */
        override_args = PyTuple_New(len + 3);
        if (override_args == NULL) {
            goto fail;
        }
        Py_INCREF(ufunc);
        PyTuple_SET_ITEM(override_args, 1, (PyObject *)ufunc);
        Py_INCREF(method_name);
        PyTuple_SET_ITEM(override_args, 2, method_name);
        for (int i = 0; i < len; i++) {
            PyObject *item = PyTuple_GET_ITEM(in_args, i);

            Py_INCREF(item);
            PyTuple_SET_ITEM(override_args, i + 3, item);
        }

        /* Check if there is a method left to call */
        if (!override_obj) {
            /* No acceptable override found. */
            static PyObject *errmsg_formatter = NULL;
            PyObject *errmsg;

            npy_cache_import("numpy.core._internal",
                             "array_ufunc_errmsg_formatter",
                             &errmsg_formatter);

            if (errmsg_formatter != NULL) {
                /* All tuple items must be set before use */
                Py_INCREF(Py_None);
                PyTuple_SET_ITEM(override_args, 0, Py_None);
                errmsg = PyObject_Call(errmsg_formatter, override_args,
                                       normal_kwds);
                if (errmsg != NULL) {
                    PyErr_SetObject(PyExc_TypeError, errmsg);
                    Py_DECREF(errmsg);
                }
            }
            Py_DECREF(override_args);
            goto fail;
        }

        /*
         * Set the self argument of our unbound method.
         * This also steals the reference, so no need to DECREF after.
         */
        PyTuple_SET_ITEM(override_args, 0, override_obj);
        /* Call the method */
        *result = PyObject_Call(
            override_array_ufunc, override_args, normal_kwds);
        Py_DECREF(override_array_ufunc);
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
    status = 0;
    /* Override found, return it. */
    goto cleanup;
fail:
    status = -1;
cleanup:
    for (int i = 0; i < num_override_args; i++) {
        Py_XDECREF(with_override[i]);
        Py_XDECREF(array_ufunc_methods[i]);
    }
    Py_XDECREF(method_name);
    Py_XDECREF(normal_kwds);
    return status;
}

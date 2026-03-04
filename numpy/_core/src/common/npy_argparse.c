#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_2_compat.h"
#include "npy_argparse.h"
#include "npy_import.h"

#include "arrayfunction_override.h"

#if PY_VERSION_HEX < 0x30d00b3
static PyThread_type_lock argparse_mutex;
#define LOCK_ARGPARSE_MUTEX                             \
    PyThread_acquire_lock(argparse_mutex, WAIT_LOCK)
#define UNLOCK_ARGPARSE_MUTEX                            \
    PyThread_release_lock(argparse_mutex)
#else
static PyMutex argparse_mutex = {0};
#define LOCK_ARGPARSE_MUTEX PyMutex_Lock(&argparse_mutex)
#define UNLOCK_ARGPARSE_MUTEX PyMutex_Unlock(&argparse_mutex)
#endif

NPY_NO_EXPORT int
init_argparse_mutex(void) {
#if PY_VERSION_HEX < 0x30d00b3
    argparse_mutex = PyThread_allocate_lock();
    if (argparse_mutex == NULL) {
        PyErr_NoMemory();
        return -1;
    }
#endif
    return 0;
}

/**
 * Small wrapper converting to array just like CPython does.
 *
 * We could use our own PyArray_PyIntAsInt function, but it handles floats
 * differently.
 * A disadvantage of this function compared to ``PyArg_*("i")`` code is that
 * it will not say which parameter is wrong.
 *
 * @param obj The python object to convert
 * @param value The output value
 *
 * @returns 0 on failure and 1 on success (`NPY_FAIL`, `NPY_SUCCEED`)
 */
NPY_NO_EXPORT int
PyArray_PythonPyIntFromInt(PyObject *obj, int *value)
{
    /* Pythons behaviour is to check only for float explicitly... */
    if (NPY_UNLIKELY(PyFloat_Check(obj))) {
        PyErr_SetString(PyExc_TypeError,
                        "integer argument expected, got float");
        return NPY_FAIL;
    }

    long result = PyLong_AsLong(obj);
    if (NPY_UNLIKELY((result == -1) && PyErr_Occurred())) {
        return NPY_FAIL;
    }
    if (NPY_UNLIKELY((result > INT_MAX) || (result < INT_MIN))) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return NPY_FAIL;
    }
    else {
        *value = (int)result;
        return NPY_SUCCEED;
    }
}


/**
 * Internal function to initialize keyword argument parsing.
 *
 * This does a few simple jobs:
 *
 * * Check the input for consistency to find coding errors, for example
 *   a parameter not marked with | after one marked with | (optional).
 * 2. Find the number of positional-only arguments, the number of
 *    total, required, and keyword arguments.
 * 3. Intern all keyword arguments strings to allow fast, identity based
 *    parsing and avoid string creation overhead on each call.
 *
 * @param funcname Name of the function, mainly used for errors.
 * @param cache A cache object stored statically in the parsing function
 * @param specs Array of argument specifications
 * @param nspecs Number of argument specifications
 * @return 0 on success, -1 on failure
 */
static int
initialize_keywords(const char *funcname,
        _NpyArgParserCache *cache, npy_arg_spec *specs, int nspecs) {
    int nkwargs = 0;
    int npositional_only = 0;
    int nrequired = 0;
    int npositional = 0;
    char state = '\0';

    for (int i = 0; i < nspecs; i++) {
        const char *name = specs[i].name;

        if (name == NULL) {
            PyErr_Format(PyExc_SystemError,
                    "NumPy internal error: name is NULL in %s() at "
                    "argument %d.", funcname, i);
            return -1;
        }
        if (specs[i].output == NULL) {
            PyErr_Format(PyExc_SystemError,
                    "NumPy internal error: data is NULL in %s() at "
                    "argument %d.", funcname, i);
            return -1;
        }

        if (*name == '|') {
            if (state == '$') {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: positional argument `|` "
                        "after keyword only `$` one to %s() at argument %d.",
                        funcname, i + 1);
                return -1;
            }
            state = '|';
            name++;  /* advance to actual name. */
            npositional += 1;
        }
        else if (*name == '$') {
            state = '$';
            name++;  /* advance to actual name. */
        }
        else {
            if (state != '\0') {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: non-required argument after "
                        "required | or $ one to %s() at argument %d.",
                        funcname, i + 1);
                return -1;
            }

            nrequired += 1;
            npositional += 1;
        }

        if (*name == '\0') {
            npositional_only += 1;
            /* Empty string signals positional only */
            if (state == '$' || npositional_only != npositional) {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: non-kwarg marked with $ "
                        "to %s() at argument %d or positional only following "
                        "kwarg.", funcname, i + 1);
                return -1;
            }
        }
        else {
            nkwargs += 1;
        }
    }

    if (npositional == -1) {
        npositional = nspecs;
    }

    if (nspecs > _NPY_MAX_KWARGS) {
        PyErr_Format(PyExc_SystemError,
                "NumPy internal error: function %s() has %d arguments, but "
                "the maximum is currently limited to %d for easier parsing; "
                "it can be increased by modifying `_NPY_MAX_KWARGS`.",
                funcname, nspecs, _NPY_MAX_KWARGS);
        return -1;
    }

    /*
     * Do any necessary string allocation and interning,
     * creating a caching object.
     */
    cache->nargs = nspecs;
    cache->npositional_only = npositional_only;
    cache->npositional = npositional;
    cache->nrequired = nrequired;

    /* NULL kw_strings for easier cleanup (and NULL termination) */
    memset(cache->kw_strings, 0, sizeof(PyObject *) * (nkwargs + 1));

    for (int i = 0; i < nspecs; i++) {
        const char *name = specs[i].name;

        if (*name == '|' || *name == '$') {
            name++;  /* ignore | and $ */
        }
        if (i >= npositional_only) {
            int i_kwarg = i - npositional_only;
            cache->kw_strings[i_kwarg] = PyUnicode_InternFromString(name);
            if (cache->kw_strings[i_kwarg] == NULL) {
                goto error;
            }
        }
    }

    return 0;

error:
    for (int i = 0; i < nkwargs; i++) {
        Py_XDECREF(cache->kw_strings[i]);
    }
    cache->npositional = -1;  /* not initialized */
    return -1;
}


static int
raise_incorrect_number_of_positional_args(const char *funcname,
        const _NpyArgParserCache *cache, Py_ssize_t len_args)
{
    const char *verb = (len_args == 1) ? "was" : "were";
    if (cache->npositional == cache->nrequired) {
        PyErr_Format(PyExc_TypeError,
                "%s() takes %d positional arguments but %zd %s given",
                funcname, cache->npositional, len_args, verb);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "%s() takes from %d to %d positional arguments but "
                "%zd %s given",
                funcname, cache->nrequired, cache->npositional,
                len_args, verb);
    }
    return -1;
}

static void
raise_missing_argument(const char *funcname,
        const _NpyArgParserCache *cache, int i)
{
    if (i < cache->npositional_only) {
        PyErr_Format(PyExc_TypeError,
                "%s() missing required positional argument %d",
                funcname, i);
    }
    else {
        PyObject *kw = cache->kw_strings[i - cache->npositional_only];
        PyErr_Format(PyExc_TypeError,
                "%s() missing required argument '%S' (pos %d)",
                funcname, kw, i);
    }
}


/**
 * Generic helper for argument parsing
 *
 * See macro version for an example pattern of how to use this function.
 *
 * @param funcname Function name
 * @param cache a NULL initialized persistent storage for data
 * @param args Python passed args (METH_FASTCALL)
 * @param len_args Number of arguments (not flagged)
 * @param kwnames Tuple as passed by METH_FASTCALL or NULL.
 * @param specs Array of argument specifications
 * @param nspecs Number of argument specifications
 *
 * @return Returns 0 on success and -1 on failure.
 */
NPY_NO_EXPORT int
_npy_parse_arguments(const char *funcname,
        _NpyArgParserCache *cache,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        npy_arg_spec *specs, int nspecs)
{
    if (!atomic_load_explicit((_Atomic(uint8_t) *)&cache->initialized, memory_order_acquire)) {
        LOCK_ARGPARSE_MUTEX;
        if (!atomic_load_explicit((_Atomic(uint8_t) *)&cache->initialized, memory_order_acquire)) {
            int res = initialize_keywords(funcname, cache, specs, nspecs);
            if (res < 0) {
                UNLOCK_ARGPARSE_MUTEX;
                return -1;
            }
            atomic_store_explicit((_Atomic(uint8_t) *)&cache->initialized, 1, memory_order_release);
        }
        UNLOCK_ARGPARSE_MUTEX;
    }

    if (NPY_UNLIKELY(len_args > cache->npositional)) {
        return raise_incorrect_number_of_positional_args(
                funcname, cache, len_args);
    }

    /* NOTE: Could remove the limit but too many kwargs are slow anyway. */
    PyObject *all_arguments[NPY_MAXARGS];

    for (Py_ssize_t i = 0; i < len_args; i++) {
        all_arguments[i] = args[i];
    }

    /* Without kwargs, do not iterate all converters. */
    int max_nargs = (int)len_args;
    Py_ssize_t len_kwargs = 0;

    /* If there are any kwargs, first handle them */
    if (NPY_LIKELY(kwnames != NULL)) {
        len_kwargs = PyTuple_GET_SIZE(kwnames);
        max_nargs = cache->nargs;

        for (int i = len_args; i < cache->nargs; i++) {
            all_arguments[i] = NULL;
        }

        for (Py_ssize_t i = 0; i < len_kwargs; i++) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = args[i + len_args];
            PyObject *const *name;

            /* Super-fast path, check identity: */
            for (name = cache->kw_strings; *name != NULL; name++) {
                if (*name == key) {
                    break;
                }
            }
            if (NPY_UNLIKELY(*name == NULL)) {
                /* Slow fallback, if identity checks failed for some reason */
                for (name = cache->kw_strings; *name != NULL; name++) {
                    int eq = PyObject_RichCompareBool(*name, key, Py_EQ);
                    if (eq == -1) {
                        return -1;
                    }
                    else if (eq) {
                        break;
                    }
                }
                if (NPY_UNLIKELY(*name == NULL)) {
                    /* Invalid keyword argument. */
                    PyErr_Format(PyExc_TypeError,
                            "%s() got an unexpected keyword argument '%S'",
                            funcname, key);
                    return -1;
                }
            }

             Py_ssize_t param_pos = (
                    (name - cache->kw_strings) + cache->npositional_only);

            /* There could be an identical positional argument */
            if (NPY_UNLIKELY(all_arguments[param_pos] != NULL)) {
                PyErr_Format(PyExc_TypeError,
                        "argument for %s() given by name ('%S') and position "
                        "(position %zd)", funcname, key, param_pos);
                return -1;
            }

            all_arguments[param_pos] = value;
        }
    }

    /*
     * There cannot be too many args, too many kwargs would find an
     * incorrect one above.
     */
    assert(len_args + len_kwargs <= cache->nargs);

    /* At this time `all_arguments` holds either NULLs or the objects */
    for (int i = 0; i < max_nargs; i++) {
        if (all_arguments[i] == NULL) {
            continue;
        }

        npy_arg_converter converter = (npy_arg_converter)specs[i].converter;
        void *data = specs[i].output;

        if (converter == NULL) {
            *((PyObject **) data) = all_arguments[i];
            continue;
        }
        int res = converter(all_arguments[i], data);

        if (NPY_UNLIKELY(res == NPY_SUCCEED)) {
            continue;
        }
        else if (NPY_UNLIKELY(res == NPY_FAIL)) {
            /* It is usually the users responsibility to clean up. */
            return -1;
        }
        else if (NPY_UNLIKELY(res == Py_CLEANUP_SUPPORTED)) {
            /* TODO: Implementing cleanup if/when needed should not be hard */
            PyErr_Format(PyExc_SystemError,
                    "converter cleanup of parameter %d to %s() not supported.",
                    i, funcname);
            return -1;
        }
        assert(0);
    }

    /* Required arguments are typically not passed as keyword arguments */
    if (NPY_UNLIKELY(len_args < cache->nrequired)) {
        /* (PyArg_* also does this after the actual parsing is finished) */
        if (NPY_UNLIKELY(max_nargs < cache->nrequired)) {
            raise_missing_argument(funcname, cache, max_nargs);
            return -1;
        }
        for (int i = 0; i < cache->nrequired; i++) {
            if (NPY_UNLIKELY(all_arguments[i] == NULL)) {
                raise_missing_argument(funcname, cache, i);
                return -1;
            }
        }
    }

    return 0;
}

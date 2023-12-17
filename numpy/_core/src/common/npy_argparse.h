#ifndef NUMPY_CORE_SRC_COMMON_NPY_ARGPARSE_H
#define NUMPY_CORE_SRC_COMMON_NPY_ARGPARSE_H

#include <Python.h>
#include "numpy/ndarraytypes.h"

/*
 * This file defines macros to help with keyword argument parsing.
 * This solves two issues as of now:
 *   1. Pythons C-API PyArg_* keyword argument parsers are slow, due to
 *      not caching the strings they use.
 *   2. It allows the use of METH_ARGPARSE (and `tp_vectorcall`)
 *      when available in Python, which removes a large chunk of overhead.
 *
 * Internally CPython achieves similar things by using a code generator
 * argument clinic. NumPy may well decide to use argument clinic or a different
 * solution in the future.
 */

NPY_NO_EXPORT int
PyArray_PythonPyIntFromInt(PyObject *obj, int *value);


#define _NPY_MAX_KWARGS 15

typedef struct {
    int npositional;
    int nargs;
    int npositional_only;
    int nrequired;
    /* Null terminated list of keyword argument name strings */
    PyObject *kw_strings[_NPY_MAX_KWARGS+1];
} _NpyArgParserCache;


/*
 * The sole purpose of this macro is to hide the argument parsing cache.
 * Since this cache must be static, this also removes a source of error.
 */
#define NPY_PREPARE_ARGPARSER static _NpyArgParserCache __argparse_cache = {-1}

/**
 * Macro to help with argument parsing.
 *
 * The pattern for using this macro is by defining the method as:
 *
 * @code
 * static PyObject *
 * my_method(PyObject *self,
 *         PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
 * {
 *     NPY_PREPARE_ARGPARSER;
 *
 *     PyObject *argument1, *argument3;
 *     int argument2 = -1;
 *     if (npy_parse_arguments("method", args, len_args, kwnames),
 *                "argument1", NULL, &argument1,
 *                "|argument2", &PyArray_PythonPyIntFromInt, &argument2,
 *                "$argument3", NULL, &argument3,
 *                NULL, NULL, NULL) < 0) {
 *          return NULL;
 *      }
 * }
 * @endcode
 *
 * The `NPY_PREPARE_ARGPARSER` macro sets up a static cache variable necessary
 * to hold data for speeding up the parsing. `npy_parse_arguments` must be
 * used in cunjunction with the macro defined in the same scope.
 * (No two `npy_parse_arguments` may share a single `NPY_PREPARE_ARGPARSER`.)
 *
 * @param funcname
 * @param args Python passed args (METH_FASTCALL)
 * @param len_args Number of arguments (not flagged)
 * @param kwnames Tuple as passed by METH_FASTCALL or NULL.
 * @param ... List of arguments must be param1_name, param1_converter,
 *            *param1_outvalue, param2_name, ..., NULL, NULL, NULL.
 *            Where name is ``char *``, ``converter`` a python converter
 *            function or NULL and ``outvalue`` is the ``void *`` passed to
 *            the converter (holding the converted data or a borrowed
 *            reference if converter is NULL).
 *
 * @return Returns 0 on success and -1 on failure.
 */
NPY_NO_EXPORT int
_npy_parse_arguments(const char *funcname,
        /* cache_ptr is a NULL initialized persistent storage for data */
        _NpyArgParserCache *cache_ptr,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        /* va_list is NULL, NULL, NULL terminated: name, converter, value */
        ...) NPY_GCC_NONNULL(1);

#define npy_parse_arguments(funcname, args, len_args, kwnames, ...)      \
        _npy_parse_arguments(funcname, &__argparse_cache,                \
                args, len_args, kwnames, __VA_ARGS__)

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_ARGPARSE_H */

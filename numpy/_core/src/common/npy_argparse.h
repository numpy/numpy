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

#define _NPY_MAX_KWARGS 14

typedef int (*npy_arg_converter)(PyObject *, void *);

typedef struct {
    const char *name;
    void *converter;
    void *output;
} npy_arg_spec;

typedef struct {
    int npositional;
    int nargs;
    int npositional_only;
    int nrequired;
    npy_uint8 initialized;
    /* Null terminated list of keyword argument name strings */
    PyObject *kw_strings[_NPY_MAX_KWARGS+1];
} _NpyArgParserCache;

NPY_NO_EXPORT int init_argparse_mutex(void);

/*
 * The sole purpose of this macro is to hide the argument parsing cache.
 * Since this cache must be static, this also removes a source of error.
 */
#define NPY_PREPARE_ARGPARSER static _NpyArgParserCache __argparse_cache;

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
 *     if (npy_parse_arguments("method", args, len_args, kwnames,
 *                {"argument1", NULL, &argument1},
 *                {"|argument2", &PyArray_PythonPyIntFromInt, &argument2},
 *                {"$argument3", NULL, &argument3}) < 0) {
 *          return NULL;
 *      }
 * }
 * @endcode
 *
 * The `NPY_PREPARE_ARGPARSER` macro sets up a static cache variable necessary
 * to hold data for speeding up the parsing. `npy_parse_arguments` must be
 * used in conjunction with the macro defined in the same scope.
 * (No two `npy_parse_arguments` may share a single `NPY_PREPARE_ARGPARSER`.)
 *
 * @param funcname Function name
 * @param args Python passed args (METH_FASTCALL)
 * @param len_args Number of arguments (not flagged)
 * @param kwnames Tuple as passed by METH_FASTCALL or NULL.
 * @param ... List of argument specs as {name, converter, outvalue} structs.
 *            Where name is ``const char *``, ``converter`` a python converter
 *            function pointer or NULL and ``outvalue`` is the ``void *``
 *            passed to the converter (holding the converted data or a
 *            borrowed reference if converter is NULL).
 *
 * @return Returns 0 on success and -1 on failure.
 */
NPY_NO_EXPORT int
_npy_parse_arguments(const char *funcname,
        _NpyArgParserCache *cache,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        npy_arg_spec *specs, int nspecs) NPY_GCC_NONNULL(1);

#ifdef __cplusplus
#define npy_parse_arguments(funcname, args, len_args, kwnames, ...)       \
        [&]() -> int {                                                     \
            npy_arg_spec _npy_specs_[] = {__VA_ARGS__};                   \
            return _npy_parse_arguments(funcname, &__argparse_cache,       \
                    args, len_args, kwnames,                               \
                    _npy_specs_,                                           \
                    (int)(sizeof(_npy_specs_) / sizeof(npy_arg_spec)));    \
        }()
#else
#define npy_parse_arguments(funcname, args, len_args, kwnames, ...)       \
        _npy_parse_arguments(funcname, &__argparse_cache,                 \
                args, len_args, kwnames,                                  \
                (npy_arg_spec[]){__VA_ARGS__},                            \
                (int)(sizeof((npy_arg_spec[]){__VA_ARGS__})               \
                      / sizeof(npy_arg_spec)))
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_ARGPARSE_H */

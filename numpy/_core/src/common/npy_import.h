#ifndef NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_

#include <Python.h>

#include "numpy/npy_common.h"
#include "npy_atomic.h"

/*
 * Holds a cached PyObject where the cache is initialized via a
 * runtime import. The cache is only filled once.
 */

typedef struct npy_runtime_import {
    npy_uint8 initialized;
    PyObject *obj;
} npy_runtime_import;

/*
 * Cached references to objects obtained via an import. All of these are
 * can be initialized at any time by npy_cache_import_runtime.
 */
typedef struct npy_runtime_imports_struct {
    PyThread_type_lock import_mutex;
    npy_runtime_import _add_dtype_helper;
    npy_runtime_import _all;
    npy_runtime_import _amax;
    npy_runtime_import _amin;
    npy_runtime_import _any;
    npy_runtime_import array_function_errmsg_formatter;
    npy_runtime_import array_ufunc_errmsg_formatter;
    npy_runtime_import _clip;
    npy_runtime_import _commastring;
    npy_runtime_import _convert_to_stringdtype_kwargs;
    npy_runtime_import _default_array_repr;
    npy_runtime_import _default_array_str;
    npy_runtime_import _dump;
    npy_runtime_import _dumps;
    npy_runtime_import _getfield_is_safe;
    npy_runtime_import internal_gcd_func;
    npy_runtime_import _mean;
    npy_runtime_import NO_NEP50_WARNING;
    npy_runtime_import npy_ctypes_check;
    npy_runtime_import numpy_matrix;
    npy_runtime_import _prod;
    npy_runtime_import _promote_fields;
    npy_runtime_import _std;
    npy_runtime_import _sum;
    npy_runtime_import _ufunc_doc_signature_formatter;
    npy_runtime_import _var;
    npy_runtime_import _view_is_safe;
    npy_runtime_import _void_scalar_to_string;
} npy_runtime_imports_struct;

NPY_VISIBILITY_HIDDEN extern npy_runtime_imports_struct npy_runtime_imports;

/*! \brief Import a Python object.

 * This function imports the Python function specified by
 * \a module and \a function, increments its reference count, and returns
 * the result. On error, returns NULL.
 *
 * @param module Absolute module name.
 * @param attr module attribute to cache.
 */
static inline PyObject*
npy_import(const char *module, const char *attr)
{
    PyObject *ret = NULL;
    PyObject *mod = PyImport_ImportModule(module);

    if (mod != NULL) {
        ret = PyObject_GetAttrString(mod, attr);
        Py_DECREF(mod);
    }
    return ret;
}

/*! \brief Fetch and cache Python object at runtime.
 *
 * Import a Python function and cache it for use. The function checks if
 * cache is NULL, and if not NULL imports the Python function specified by
 * \a module and \a function, increments its reference count, and stores
 * the result in \a cache. Usually \a cache will be a static variable and
 * should be initialized to NULL. On error \a cache will contain NULL on
 * exit,
 *
 * @param module Absolute module name.
 * @param attr module attribute to cache.
 * @param cache Storage location for imported function.
 */
static inline int
npy_cache_import_runtime(const char *module, const char *attr, npy_runtime_import *cache) {
    if (cache->initialized) {
        return 0;
    }
    else {
        if (!npy_atomic_load_uint8(&cache->initialized)) {
            PyThread_acquire_lock(npy_runtime_imports.import_mutex, WAIT_LOCK);
            if (!cache->initialized) {
                cache->obj = npy_import(module, attr);
                cache->initialized = 1;
            }
            PyThread_release_lock(npy_runtime_imports.import_mutex);
        }
    }
    if (cache->obj == NULL) {
        return -1;
    }
    return 0;    
}

NPY_NO_EXPORT int
init_import_mutex(void);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_

#include <Python.h>

#include "numpy/npy_common.h"
#include "npy_atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cached references to objects obtained via an import. All of these are
 * can be initialized at any time by npy_cache_import_runtime.
 */
typedef struct npy_runtime_imports_struct {
#if PY_VERSION_HEX < 0x30d00b3
    PyThread_type_lock import_mutex;
#else
    PyMutex import_mutex;
#endif
    PyObject *_add_dtype_helper;
    PyObject *_all;
    PyObject *_amax;
    PyObject *_amin;
    PyObject *_any;
    PyObject *array_function_errmsg_formatter;
    PyObject *array_ufunc_errmsg_formatter;
    PyObject *_clip;
    PyObject *_commastring;
    PyObject *_convert_to_stringdtype_kwargs;
    PyObject *_default_array_repr;
    PyObject *_default_array_str;
    PyObject *_dump;
    PyObject *_dumps;
    PyObject *_getfield_is_safe;
    PyObject *internal_gcd_func;
    PyObject *_mean;
    PyObject *NO_NEP50_WARNING;
    PyObject *npy_ctypes_check;
    PyObject *numpy_matrix;
    PyObject *_prod;
    PyObject *_promote_fields;
    PyObject *_std;
    PyObject *_sum;
    PyObject *_ufunc_doc_signature_formatter;
    PyObject *_usefields;
    PyObject *_var;
    PyObject *_view_is_safe;
    PyObject *_void_scalar_to_string;
    PyObject *sort;
    PyObject *argsort;
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
 * @param obj Storage location for imported function.
 */
static inline int
npy_cache_import_runtime(const char *module, const char *attr, PyObject **obj) {
    if (!npy_atomic_load_ptr(obj)) {
        PyObject* value = npy_import(module, attr);
        if (value == NULL) {
            return -1;
        }
#if PY_VERSION_HEX < 0x30d00b3
        PyThread_acquire_lock(npy_runtime_imports.import_mutex, WAIT_LOCK);
#else
        PyMutex_Lock(&npy_runtime_imports.import_mutex);
#endif
        if (!npy_atomic_load_ptr(obj)) {
            npy_atomic_store_ptr(obj, Py_NewRef(value));
        }
#if PY_VERSION_HEX < 0x30d00b3
        PyThread_release_lock(npy_runtime_imports.import_mutex);
#else
        PyMutex_Unlock(&npy_runtime_imports.import_mutex);
#endif
        Py_DECREF(value);
    }
    return 0;
}

NPY_NO_EXPORT int
init_import_mutex(void);

/*! \brief Import a Python object from an entry point string.

 * The name should be of the form "(module ':')? (object '.')* attr".
 * If no module is present, it is assumed to be "numpy".
 * On error, returns NULL.
 */
NPY_NO_EXPORT PyObject*
npy_import_entry_point(const char *entry_point);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_

#include <Python.h>

#include "numpy/npy_common.h"
#include "module_state_fields.h"

#ifdef __cplusplus
extern "C" {
#endif

#if PY_VERSION_HEX < 0x30d00b3 || defined(Py_LIMITED_API)
    #define NPY_USE_LEGACY_LOCK 1
#endif

/*
 * Cached references to objects obtained via an import. All of these are
 * can be initialized at any time by npy_cache_import_runtime.
 */
typedef struct npy_runtime_imports_struct {
    NPY_DECLARE_PYOBJECT_FIELDS(NPY_RUNTIME_IMPORTS_FIELDS)
} npy_runtime_imports_struct;

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

NPY_NO_EXPORT int
init_import_mutex(void);

/*! \brief Import a Python object from an entry point string.

 * The name should be of the form "(module ':')? (object '.')* attr".
 * If no module is present, it is assumed to be "numpy".
 * On error, returns NULL.
 */
NPY_NO_EXPORT PyObject*
npy_import_entry_point(const char *entry_point);


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
NPY_NO_EXPORT int
npy_cache_import_runtime(const char *module, const char *attr, PyObject **obj);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_

#include <Python.h>

/*! \brief Fetch and cache Python function.
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
static inline void
npy_cache_import(const char *module, const char *attr, PyObject **cache)
{
    if (NPY_UNLIKELY(*cache == NULL)) {
        PyObject *mod = PyImport_ImportModule(module);

        if (mod != NULL) {
            *cache = PyObject_GetAttrString(mod, attr);
            Py_DECREF(mod);
        }
    }
}

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_IMPORT_H_ */

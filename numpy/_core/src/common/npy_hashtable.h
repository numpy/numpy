#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


#ifdef __cplusplus
extern "C" {
#endif

struct buckets;

typedef struct {
    int key_len;             /* number of identities used */
    struct buckets *buckets; /* current buckets */
#ifdef Py_GIL_DISABLED
    PyMutex mutex;
#endif
} PyArrayIdentityHash;


NPY_NO_EXPORT int
PyArrayIdentityHash_SetItemDefault(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *default_value, PyObject **result);

NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash *tb, PyObject *const *key);

NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len);

NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_ */

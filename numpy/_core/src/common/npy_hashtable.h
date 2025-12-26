#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


#ifdef __cplusplus
extern "C" {
#endif

struct buckets {
    struct buckets *prev; /* linked list of old buckets */
    npy_intp size;        /* current size */
    npy_intp nelem;       /* number of elements */
    PyObject *array[];
};

typedef struct {
    int key_len;  /* number of identities used */
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    struct buckets *buckets;
#ifdef Py_GIL_DISABLED
    PyMutex mutex;
#endif
} PyArrayIdentityHash;


NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace);

NPY_NO_EXPORT int
PyArrayIdentityHash_SetItemLockHeld(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace);

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

#ifndef NPY_RWLOCK_H_
#define NPY_RWLOCK_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PyRWMutex {
    PyMutex reader_lock;
    PyMutex writer_lock;
    uint32_t reader_count;
    unsigned long writer_id;
    unsigned long level;
} PyRWMutex;

NPY_NO_EXPORT void PyRWMutex_Lock(PyRWMutex *rwmutex);
NPY_NO_EXPORT void PyRWMutex_Unlock(PyRWMutex *rwmutex);
NPY_NO_EXPORT void PyRWMutex_RLock(PyRWMutex *rwmutex);
NPY_NO_EXPORT void PyRWMutex_RUnlock(PyRWMutex *rwmutex);

#ifdef __cplusplus
}
#endif

#endif /* NPY_RWLOCK_H_ */
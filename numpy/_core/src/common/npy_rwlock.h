#ifndef NPY_RWLOCK_H_
#define NPY_RWLOCK_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef Py_GIL_DISABLED

/*
    A read-write mutex implemented using PyMutex.
    It allows multiple readers or one writer at a time.
    If the current thread holds the write lock, it can acquire
    recursive read or write locks.

    As this is implemented using PyMutex, it automatically
    releases the GIL or thread state when locking blocks.

    Structure members:
    - reader_lock: Mutex to protect reader_count.
    - writer_lock: Mutex to protect write access.
    - reader_count: Number of active readers.
    - writer_id: Thread ID of the writer holding the lock (0 if no writer).
    - level: Recursion level for the current thread holding the write lock.
*/

typedef struct NPyRWMutex {
    PyMutex reader_lock;
    PyMutex writer_lock;
    uint32_t reader_count;
    uint32_t level;
    unsigned long writer_id;
} NPyRWMutex;

// Write lock the RWMutex
NPY_NO_EXPORT void NPyRWMutex_Lock(NPyRWMutex *rwmutex);
// Write unlock the RWMutex
NPY_NO_EXPORT void NPyRWMutex_Unlock(NPyRWMutex *rwmutex);
// Read lock the RWMutex
NPY_NO_EXPORT void NPyRWMutex_RLock(NPyRWMutex *rwmutex);
// Read unlock the RWMutex
NPY_NO_EXPORT void NPyRWMutex_RUnlock(NPyRWMutex *rwmutex);

#endif /* Py_GIL_DISABLED */

#ifdef __cplusplus
}
#endif

#endif /* NPY_RWLOCK_H_ */

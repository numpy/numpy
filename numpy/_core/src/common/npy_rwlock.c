#include "npy_rwlock.h"
#include "npy_atomic.h"

#ifdef Py_GIL_DISABLED

// PyMutex_IsLocked is available only in Python 3.14+
#ifndef PyMutex_IsLocked
#define PyMutex_IsLocked(m) (1)
#endif

NPY_NO_EXPORT void
PyRWMutex_Lock(PyRWMutex *rwmutex)
{
    unsigned long thread_id = PyThread_get_thread_ident();
    unsigned long writer_id = npy_atomic_load_ulong(&rwmutex->writer_id);
    if (writer_id == thread_id) {
        // If the current thread already holds the write lock, allow recursive write lock
        rwmutex->level++;
        assert(PyMutex_IsLocked(&rwmutex->writer_lock));
        return;
    }
    PyMutex_Lock(&rwmutex->writer_lock);
    assert(rwmutex->writer_id == 0);
    npy_atomic_store_ulong(&rwmutex->writer_id, thread_id);
}

NPY_NO_EXPORT void
PyRWMutex_Unlock(PyRWMutex *rwmutex)
{
    assert(PyMutex_IsLocked(&rwmutex->writer_lock));
    assert(rwmutex->writer_id == PyThread_get_thread_ident());
    if (rwmutex->level > 0) {
        rwmutex->level--;
        return;
    }
    npy_atomic_store_ulong(&rwmutex->writer_id, 0);
    PyMutex_Unlock(&rwmutex->writer_lock);
}

NPY_NO_EXPORT void
PyRWMutex_RLock(PyRWMutex *rwmutex)
{
    unsigned long thread_id = PyThread_get_thread_ident();
    unsigned long writer_id = npy_atomic_load_ulong(&rwmutex->writer_id);
    // If current thread holds the write lock, allow recursive read lock
    if (writer_id == thread_id) {
        rwmutex->level++;
        assert(PyMutex_IsLocked(&rwmutex->writer_lock));
        return;
    }

    PyMutex_Lock(&rwmutex->reader_lock);
    rwmutex->reader_count++;
    if (rwmutex->reader_count == 1) {
        // First reader acquires the write lock to block writers
        PyMutex_Lock(&rwmutex->writer_lock);
    }
    assert(PyMutex_IsLocked(&rwmutex->writer_lock));
    assert(rwmutex->writer_id == 0);
    PyMutex_Unlock(&rwmutex->reader_lock);
}

NPY_NO_EXPORT void
PyRWMutex_RUnlock(PyRWMutex *rwmutex)
{
    assert(PyMutex_IsLocked(&rwmutex->writer_lock));
    assert(rwmutex->writer_id == 0 || rwmutex->writer_id == PyThread_get_thread_ident());
    if (rwmutex->level > 0) {
        assert(rwmutex->writer_id == PyThread_get_thread_ident());
        rwmutex->level--;
        return;
    }
    PyMutex_Lock(&rwmutex->reader_lock);
    rwmutex->reader_count--;
    if (rwmutex->reader_count == 0) {
        // Last reader releases the writer lock
        assert(rwmutex->writer_id == 0);
        PyMutex_Unlock(&rwmutex->writer_lock);
    }
    PyMutex_Unlock(&rwmutex->reader_lock);
}

#endif /* Py_GIL_DISABLED */
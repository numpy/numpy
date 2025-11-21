#include "npy_rwlock.h"

#include <stdatomic.h>
#include <stdbool.h>

NPY_NO_EXPORT void
PyRWMutex_Lock(PyRWMutex *rwmutex)
{
    unsigned long thread_id = PyThread_get_thread_ident();
    unsigned long writer_id = atomic_load_explicit((_Atomic unsigned long *)&rwmutex->writer_id, memory_order_acquire);
    if (writer_id == thread_id) {
        // If the current thread already holds the write lock, allow recursive write lock
        rwmutex->level++;
        return;
    }
    PyMutex_Lock(&rwmutex->writer_lock);
    atomic_store_explicit((_Atomic unsigned long *)&rwmutex->writer_id, thread_id, memory_order_release);
}

NPY_NO_EXPORT void
PyRWMutex_Unlock(PyRWMutex *rwmutex)
{
    unsigned long thread_id = PyThread_get_thread_ident();
    if (rwmutex->writer_id != thread_id) {
        Py_FatalError("Attempt to unlock a RWMutex not owned by the thread");
    }
    if (rwmutex->level > 0) {
        rwmutex->level--;
        return;
    }
    atomic_store_explicit((_Atomic unsigned long *)&rwmutex->writer_id, 0, memory_order_release);
    PyMutex_Unlock(&rwmutex->writer_lock);
}

NPY_NO_EXPORT void
PyRWMutex_RLock(PyRWMutex *rwmutex)
{
    unsigned long thread_id = PyThread_get_thread_ident();
    unsigned long writer_id = atomic_load_explicit((_Atomic unsigned long *)&rwmutex->writer_id, memory_order_acquire);
    // If current thread holds the write lock, allow recursive read lock
    if (writer_id == thread_id) {
        rwmutex->level++;
        return;
    }

    PyMutex_Lock(&rwmutex->reader_lock);
    rwmutex->reader_count++;
    if (rwmutex->reader_count == 1) {
        // First reader acquires the writer lock to block writers
        PyMutex_Lock(&rwmutex->writer_lock);
        // zero means locked by reader
        atomic_store_explicit((_Atomic unsigned long *)&rwmutex->writer_id, 0, memory_order_release);
    }
    PyMutex_Unlock(&rwmutex->reader_lock);
}

NPY_NO_EXPORT void
PyRWMutex_RUnlock(PyRWMutex *rwmutex)
{
    if (rwmutex->level > 0) {
        rwmutex->level--;
        return;
    }
    PyMutex_Lock(&rwmutex->reader_lock);
    rwmutex->reader_count--;
    if (rwmutex->reader_count == 0) {
        // Last reader releases the writer lock
        atomic_store_explicit((_Atomic unsigned long *)&rwmutex->writer_id, 0, memory_order_release);
        PyMutex_Unlock(&rwmutex->writer_lock);
    }
    PyMutex_Unlock(&rwmutex->reader_lock);
}
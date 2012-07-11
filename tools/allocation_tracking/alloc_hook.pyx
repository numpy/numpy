# A cython wrapper for using python functions as callbacks for
# PyDataMem_SetEventHook.

cimport numpy as np

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void *PyLong_AsVoidPtr(object)

ctypedef void PyDataMem_EventHookFunc(void *inp, void *outp, size_t size,
                                      void *user_data)
cdef extern from "numpy/arrayobject.h":
    PyDataMem_EventHookFunc * \
        PyDataMem_SetEventHook(PyDataMem_EventHookFunc *newhook,
                               void *user_data, void **old_data)

np.import_array()

cdef void pyhook(void *old, void *new, size_t size, void *user_data):
    cdef object pyfunc = <object> user_data
    pyfunc(PyLong_FromVoidPtr(old),
           PyLong_FromVoidPtr(new),
           size)

class NumpyAllocHook(object):
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        cdef void *old_hook, *old_data
        old_hook = <void *> \
            PyDataMem_SetEventHook(<PyDataMem_EventHookFunc *> pyhook,
                                    <void *> self.callback,
                                    <void **> &old_data)
        self.old_hook = PyLong_FromVoidPtr(old_hook)
        self.old_data = PyLong_FromVoidPtr(old_data)

    def __exit__(self):
        PyDataMem_SetEventHook(<PyDataMem_EventHookFunc *> \
                                    PyLong_AsVoidPtr(self.old_hook),
                                <void *> PyLong_AsVoidPtr(self.old_data),
                                <void **> 0)

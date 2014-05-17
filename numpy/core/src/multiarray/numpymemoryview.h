#ifndef _NPY_PRIVATE_NUMPYMEMORYVIEW_H_
#define _NPY_PRIVATE_NUMPYMEMORYVIEW_H_

/*
 * Memoryview is introduced to 2.x series only in 2.7, so for supporting 2.6,
 * we need to have a minimal implementation here.
 */
#if PY_VERSION_HEX < 0x02070000

typedef struct {
    PyObject_HEAD
    PyObject *base;
    Py_buffer view;
} PyMemorySimpleViewObject;

NPY_NO_EXPORT PyObject *
PyMemorySimpleView_FromObject(PyObject *base);

#define PyMemorySimpleView_GET_BUFFER(op) (&((PyMemorySimpleViewObject *)(op))->view)

#define PyMemoryView_FromObject PyMemorySimpleView_FromObject
#define PyMemoryView_GET_BUFFER PyMemorySimpleView_GET_BUFFER

#endif

NPY_NO_EXPORT int
_numpymemoryview_init(PyObject **typeobject);

#endif

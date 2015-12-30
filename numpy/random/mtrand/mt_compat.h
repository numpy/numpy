/*
 * This is a convenience header file providing compatibility utilities
 * for supporting Python 2 and Python 3 in the same code base.
 *
 * It can be removed when Python 2.6 is dropped as PyCapsule is available
 * in both Python 3.1+ and Python 2.7.
 */

#ifndef _MT_COMPAT_H_
#define _MT_COMPAT_H_

#include <Python.h>
#include <numpy/npy_common.h>

#ifdef __cplusplus
extern "C" {
#endif


/*
 * PyCObject functions adapted to PyCapsules.
 *
 * The main job here is to get rid of the improved error handling
 * of PyCapsules. It's a shame...
 */
#if PY_VERSION_HEX >= 0x03000000

static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *))
{
    PyObject *ret = PyCapsule_New(ptr, NULL, dtor);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static NPY_INLINE void *
NpyCapsule_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

#else

static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(void *))
{
    return PyCObject_FromVoidPtr(ptr, dtor);
}

static NPY_INLINE void *
NpyCapsule_AsVoidPtr(PyObject *ptr)
{
    return PyCObject_AsVoidPtr(ptr);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* _COMPAT_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"

#ifndef Py_LIMITED_API
#include "pythoncapi-compat/pythoncapi_compat.h"
#else
static inline PyObject *
PyTuple_FromArray(PyObject *const *array, Py_ssize_t size)
{
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < size; i++) {
        if (PyTuple_SetItem(tuple, i, Py_NewRef(array[i])) < 0) {
            Py_DECREF(tuple);
            return NULL;
        }
    }
    return tuple;
}

static inline int
PyLong_IsZero(PyObject *obj)
{
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "expected int, got %T", obj);
        return -1;
    }
    int overflow;
    return PyLong_AsLongAndOverflow(obj, &overflow) == 0;
}

/*
 * TODO: remove when all supported Pythons define these in the Limited API,
 * see https://github.com/capi-workgroup/decisions/issues/110
 */
#ifndef Py_SETREF
#if defined(__GNUC__) || defined(__clang__)
#define NPY__SETREF(dst, src, decref)                                   \
    do {                                                                \
        __typeof__(dst) *_npy_dst_ptr = &(dst);                         \
        __typeof__(dst) _npy_old_dst = *_npy_dst_ptr;                   \
        *_npy_dst_ptr = (src);                                          \
        decref((PyObject *)_npy_old_dst);                               \
    } while (0)
#else
#include <string.h>
#define NPY__SETREF(dst, src, decref)                                   \
    do {                                                                \
        PyObject **_npy_dst_ptr = (PyObject **)&(dst);                  \
        PyObject *_npy_old_dst = *_npy_dst_ptr;                         \
        PyObject *_npy_src = (PyObject *)(src);                         \
        memcpy(_npy_dst_ptr, &_npy_src, sizeof(PyObject *));            \
        decref(_npy_old_dst);                                           \
    } while (0)
#endif
#define Py_SETREF(dst, src) NPY__SETREF(dst, src, Py_DECREF)
#define Py_XSETREF(dst, src) NPY__SETREF(dst, src, Py_XDECREF)
#endif

#define PyTuple_GET_SIZE(op) PyTuple_Size(op)
#define PyTuple_GET_ITEM(op, i) PyTuple_GetItem(op, i)
#define PyTuple_SET_ITEM(op, i, v) PyTuple_SetItem(op, i, v)
#define PyList_GET_SIZE(op) PyList_Size(op)
#define PyList_GET_ITEM(op, i) PyList_GetItem(op, i)
#define PyList_SET_ITEM(op, i, v) PyList_SetItem(op, i, v)
/* Python.h defines these before 3.14 */
#ifndef PySequence_Fast_GET_SIZE
#define PySequence_Fast_GET_SIZE(o)                                     \
    (PyList_Check(o) ? PyList_GET_SIZE(o) : PyTuple_GET_SIZE(o))
#define PySequence_Fast_GET_ITEM(o, i)                                  \
    (PyList_Check(o) ? PyList_GET_ITEM(o, i) : PyTuple_GET_ITEM(o, i))
#endif
#define PyBytes_AS_STRING(op) PyBytes_AsString(op)
#define PyBytes_GET_SIZE(op) PyBytes_Size(op)
#define PyFloat_AS_DOUBLE(op) PyFloat_AsDouble(op)
#define PyDict_GET_SIZE(op) PyDict_Size(op)

/*
 * No-ops with the GIL, as CPython defines them. From 3.15 the Limited API
 * versions call functions that are not in the stable ABI before 3.15.
 */
#ifndef Py_GIL_DISABLED
#undef Py_BEGIN_CRITICAL_SECTION
#undef Py_END_CRITICAL_SECTION
#define Py_BEGIN_CRITICAL_SECTION(op) {
#define Py_END_CRITICAL_SECTION() }
#endif
#endif  /* Py_LIMITED_API */

#define Npy_HashDouble _Py_HashDouble

#ifdef Py_GIL_DISABLED
// Specialized version of critical section locking to safely use
// PySequence_Fast APIs without the GIL. For performance, the argument *to*
// PySequence_Fast() is provided to the macro, not the *result* of
// PySequence_Fast(), which would require an extra test to determine if the
// lock must be acquired.
//
// These are tweaked versions of macros defined in CPython in
// pycore_critical_section.h, originally added in CPython commit baf347d91643.
// They should behave identically to the versions in CPython. Once the
// macros are expanded, the only difference relative to those versions is the
// use of public C API symbols that are equivalent to the ones used in the
// corresponding CPython definitions.
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original)              \
    {                                                                   \
        PyObject *_orig_seq = (PyObject *)(original);                   \
        const int _should_lock_cs =                                     \
                PyList_CheckExact(_orig_seq);                           \
        PyCriticalSection _cs_fast;                                     \
        if (_should_lock_cs) {                                          \
            PyCriticalSection_Begin(&_cs_fast, _orig_seq);              \
        }
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST()                        \
        if (_should_lock_cs) {                                          \
            PyCriticalSection_End(&_cs_fast);                           \
        }                                                               \
    }
#else
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original) { do { (void)(original); } while (0)
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST() }
#endif


#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */

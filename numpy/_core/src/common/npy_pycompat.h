#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"
#include "pythoncapi-compat/pythoncapi_compat.h"

#define Npy_HashDouble _Py_HashDouble

/* Like PyObject_Print(obj, fp, 0), which is not in the Limited API */
static inline int
npy_PyObject_Print(PyObject *obj, FILE *fp)
{
    if (obj == NULL) {
        fputs("<nil>", fp);
        return 0;
    }
    PyObject *repr = PyObject_Repr(obj);
    if (repr == NULL) {
        return -1;
    }
    Py_ssize_t size;
    const char *str = PyUnicode_AsUTF8AndSize(repr, &size);
    if (str != NULL) {
        fwrite(str, 1, (size_t)size, fp);
    }
    Py_DECREF(repr);
    return str == NULL ? -1 : 0;
}

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

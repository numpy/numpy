#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"
#include "pythoncapi-compat/pythoncapi_compat.h"

#define Npy_HashDouble _Py_HashDouble

// These are slightly tweaked versions of macros defined in CPython in
// pycore_critical_section.h, originally added in CPython commit baf347d91643.
//
// The tweaks are only to use public CPython C API symbols

#ifdef Py_GIL_DISABLED
// Specialized version of critical section locking to safely use
// PySequence_Fast APIs without the GIL. For performance, the argument *to*
// PySequence_Fast() is provided to the macro, not the *result* of
// PySequence_Fast(), which would require an extra test to determine if the
// lock must be acquired.
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original)  \
    PyObject *_orig_seq = (PyObject *)(original);                       \
    const int _should_lock_cs = PyList_CheckExact(_orig_seq);           \
    PyCriticalSection _cs;                                              \
    if (_should_lock_cs) {                                              \
        PyCriticalSection_Begin(&_cs, _orig_seq);                       \
    }

#define Py_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original)               \
    {                                                                   \
        NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original)
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS()    \
    if (_should_lock_cs) {                                      \
        PyCriticalSection_End(&_cs);                            \
    }
#define Py_END_CRITICAL_SECTION_SEQUENCE_FAST()                 \
        NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS()
    }
#define NPY_BEGIN_CRITICAL_SECTION_NO_BRACKETS(obj)     \
    PyCriticalSection _py_cs;                           \
    PyCriticalSection_Begin(&_py_cs, (PyObject *)(op))
#define NPY_END_CRITICAL_SECTION_NO_BRACKETS() PyCriticalSection_End(&_py_cs)

#else
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original)
#define Py_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original) {
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original)
#define Py_END_CRITICAL_SECTION_SEQUENCE_FAST() }
#define NPY_BEGIN_CRITICAL_SECTION_NO_BRACKETS(obj)
#define NPY_END_CRITICAL_SECTION_NO_BRACKETS()
#endif


#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */

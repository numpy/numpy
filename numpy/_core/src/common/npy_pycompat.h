#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"
#include "pythoncapi-compat/pythoncapi_compat.h"

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
// They're defined in terms of the NPY_*_CRITICAL_SECTION_NO_BRACKETS to avoid
// repition and should behave identically to the versions in CPython. Once the
// macros are expanded, The only difference relative to those versions is the
// use of public C API symbols that are equivalent to the ones used in the
// corresponding CPython definitions.
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original)              \
    {                                                                   \
    NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(               \
            original, npy_cs_fast)
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST()                        \
        NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(npy_cs_fast) \
    }

// These macros are more flexible than the versions in the public CPython C API,
// but that comes at a cost. Here are some differences and limitations:
//
// * cs_name is a named label for the critical section. If you must nest
//   critical sections, do *not* use the same name for multiple nesting
//   critical sections.
// * The beginning and ending macros must happen within the same scope
//   and the compiler won't necessarily enforce that.
// * The macros ending critical sections accept a named label. The label
//   must match the opening critical section.
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original, cs_name) \
    PyObject *_##cs_name##_orig_seq = (PyObject *)(original);           \
    const int _##cs_name##_should_lock_cs =                             \
            PyList_CheckExact(_##cs_name##_orig_seq);                   \
    PyCriticalSection _##cs_name;                                       \
    if (_##cs_name##_should_lock_cs) {                                  \
        PyCriticalSection_Begin(&_##cs_name, _##cs_name##_orig_seq);    \
    }
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(cs_name)     \
    if (_##cs_name##_should_lock_cs) {                                  \
        PyCriticalSection_End(&_##cs_name);                             \
    }
#else
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(original, cs_name)
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original) {
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(cs_name)
#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST() }
#endif


#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */

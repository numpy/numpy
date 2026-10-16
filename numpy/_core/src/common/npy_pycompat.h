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

#ifdef Py_LIMITED_API
/*
 * Create a `str` from `size` UCS4 characters in native byte order.
 */
static inline PyObject *
npy_PyUnicode_FromUCS4(const void *buf, Py_ssize_t size)
{
    /* A given byte order means no BOM handling; lone surrogates are kept */
    int byteorder = PY_LITTLE_ENDIAN ? -1 : 1;
    return PyUnicode_DecodeUTF32(
            (const char *)buf, size * 4, "surrogatepass", &byteorder);
}

/*
 * Whether a `str` has only ASCII characters.  This cannot fail.
 */
static inline int
npy_PyUnicode_IsASCII(PyObject *str)
{
    Py_ssize_t size;
    if (PyUnicode_AsUTF8AndSize(str, &size) == NULL) {
        /* Encoding can only fail for a `str` that is not ASCII */
        PyErr_Clear();
        return 0;
    }
    return size == PyUnicode_GetLength(str);
}
#else
#define npy_PyUnicode_FromUCS4(buf, size) \
    PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf, size)
#define npy_PyUnicode_IsASCII(str) PyUnicode_IS_ASCII(str)
#endif


#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_

#if defined(_MSC_VER)
// Suppress warn C4146: -x is valid for unsigned (wraps around)
#pragma warning(disable:4146)
#endif

#include "config.h"
#include "npy_cpu_dispatch.h" // brings NPY_HAVE_[CPU features]
#include "numpy/numpyconfig.h"
#include "numpy/utils.h"
#include "numpy/npy_os.h"

/* blocklist */

/* Disable broken functions on z/OS */
#if defined (__MVS__)

#define NPY_BLOCK_POWF
#define NPY_BLOCK_EXPF
#undef HAVE___THREAD

#endif

/* Disable broken MS math functions */
#if defined(__MINGW32_VERSION)

#define NPY_BLOCK_ATAN2
#define NPY_BLOCK_ATAN2F
#define NPY_BLOCK_ATAN2L

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif

#if defined(_MSC_VER)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CSQRT
#undef HAVE_CSQRTF
#undef HAVE_CSQRTL
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif

/* Work around MSVC compiler bug, see gh-32229 */
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER == 1951

#undef HAVE___BUILTIN_ISNAN
#undef HAVE___BUILTIN_ISINF
#undef HAVE___BUILTIN_ISFINITE

#endif

/* MSVC _hypot messes with fp precision mode on 32-bit, see gh-9567 */
#if defined(_MSC_VER) && !defined(_WIN64)

#undef HAVE_CABS
#undef HAVE_CABSF
#undef HAVE_CABSL

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif


/* Intel C for Windows uses POW for 64 bits longdouble*/
#if defined(_MSC_VER) && defined(__INTEL_COMPILER)
#if NPY_SIZEOF_LONGDOUBLE == 8
#define NPY_BLOCK_POWL
#endif
#endif /* defined(_MSC_VER) && defined(__INTEL_COMPILER) */

/* powl gives zero division warning on OS X, see gh-8307 */
#if defined(NPY_OS_DARWIN)
#define NPY_BLOCK_POWL
#endif

#ifdef __CYGWIN__
/* Loss of precision */
#undef HAVE_CASINHL
#undef HAVE_CASINH
#undef HAVE_CASINHF

/* Loss of precision */
#undef HAVE_CATANHL
#undef HAVE_CATANH
#undef HAVE_CATANHF

/* Loss of precision and branch cuts */
#undef HAVE_CATANL
#undef HAVE_CATAN
#undef HAVE_CATANF

/* Branch cuts */
#undef HAVE_CACOSHF
#undef HAVE_CACOSH

/* Branch cuts */
#undef HAVE_CSQRTF
#undef HAVE_CSQRT

/* Branch cuts and loss of precision */
#undef HAVE_CASINF
#undef HAVE_CASIN
#undef HAVE_CASINL

/* Branch cuts */
#undef HAVE_CACOSF
#undef HAVE_CACOS

/* log2(exp2(i)) off by a few eps */
#define NPY_BLOCK_LOG2

/* np.power(..., dtype=np.complex256) doesn't report overflow */
#undef HAVE_CPOWL
#undef HAVE_CEXPL

/*
 * cygwin uses newlib, which has naive implementations of the
 * complex log functions.
 */
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL

#include <cygwin/version.h>
#if CYGWIN_VERSION_DLL_MAJOR < 3003
// rather than blocklist cabsl, hypotl, modfl, sqrtl, error out
#error cygwin < 3.3 not supported, please update
#endif
#endif

/* Disable broken gnu trig functions */
#if defined(HAVE_FEATURES_H)
#include <features.h>

#if defined(__GLIBC__)
#if !__GLIBC_PREREQ(2, 18)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif  /* __GLIBC_PREREQ(2, 18) */
#else   /* defined(__GLIBC) */
/* musl linux?, see issue #25092 */

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

/*
 * musl's clog is low precision for some inputs.  As of MUSL 1.2.5,
 * the first comment in clog.c is "// FIXME".
 * See https://github.com/numpy/numpy/pull/24416#issuecomment-1678208628
 * and https://github.com/numpy/numpy/pull/24448
 */
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL

#endif  /* defined(__GLIBC) */
#endif  /* defined(HAVE_FEATURES_H) */

/* CPython macros that the Limited API does not provide */
#ifdef Py_LIMITED_API

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
 * versions call functions that older Pythons do not have.
 */
#ifndef Py_GIL_DISABLED
#undef Py_BEGIN_CRITICAL_SECTION
#undef Py_END_CRITICAL_SECTION
#define Py_BEGIN_CRITICAL_SECTION(op) {
#define Py_END_CRITICAL_SECTION() }
#endif

#endif  /* Py_LIMITED_API */

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_ */

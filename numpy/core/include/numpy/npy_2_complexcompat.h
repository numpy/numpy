/* This header is designed to be copy-pasted into downstream packages, since it provides
   a compatibility layer between the old C struct complex types and the new native C99
   complex types. */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_

#include <numpy/numpyconfig.h>

#if (defined(NPY_TARGET_VERSION) && NPY_TARGET_VERSION >= 0x00000012) || (NPY_FEATURE_VERSION >= 0x00000012)  /* Numpy 2.0 */
#include <numpy/npy_math.h>

#define NPY_CSETREALF(c, r) npy_csetrealf(c, r)
#define NPY_CSETIMAGF(c, i) npy_csetimagf(c, i)
#define NPY_CSETREAL(c, r)  npy_csetreal(c, r)
#define NPY_CSETIMAG(c, i)  npy_csetimag(c, i)
#define NPY_CSETREALL(c, r) npy_csetreall(c, r)
#define NPY_CSETIMAGL(c, i) npy_csetimagl(c, i)
#else
#define NPY_CSETREALF(c, r) (c)->real = (r)
#define NPY_CSETIMAGF(c, i) (c)->imag = (i)
#define NPY_CSETREAL(c, r)  (c)->real = (r)
#define NPY_CSETIMAG(c, i)  (c)->imag = (i)
#define NPY_CSETREALL(c, r) (c)->real = (r)
#define NPY_CSETIMAGL(c, i) (c)->imag = (i)
#endif

#endif

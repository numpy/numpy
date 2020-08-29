#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h"
#include "numpy/numpyconfig.h"
#include "numpy/npy_cpu.h"
#include "numpy/npy_os.h"

/* blocklist */

/* Disable broken Sun Workshop Pro math functions */
#ifdef __SUNPRO_C

#undef HAVE_ATAN2
#undef HAVE_ATAN2F
#undef HAVE_ATAN2L

#endif

/* Disable broken functions on z/OS */
#if defined (__MVS__)

#undef HAVE_POWF
#undef HAVE_EXPF
#undef HAVE___THREAD

#endif

/* Disable broken MS math functions */
#if (defined(_MSC_VER) && (_MSC_VER < 1900)) || defined(__MINGW32_VERSION)

#undef HAVE_ATAN2
#undef HAVE_ATAN2F
#undef HAVE_ATAN2L

#undef HAVE_HYPOT
#undef HAVE_HYPOTF
#undef HAVE_HYPOTL

#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1900)

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

/* MSVC _hypot messes with fp precision mode on 32-bit, see gh-9567 */
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(_WIN64)

#undef HAVE_CABS
#undef HAVE_CABSF
#undef HAVE_CABSL

#undef HAVE_HYPOT
#undef HAVE_HYPOTF
#undef HAVE_HYPOTL

#endif


/* Intel C for Windows uses POW for 64 bits longdouble*/
#if defined(_MSC_VER) && defined(__INTEL_COMPILER)
#if defined(HAVE_POWL) && (NPY_SIZEOF_LONGDOUBLE == 8)
#undef HAVE_POWL
#endif
#endif /* defined(_MSC_VER) && defined(__INTEL_COMPILER) */

/* Disable functions that give undesired floating-point outputs on
   Cygwin. */
#ifdef __CYGWIN__

#undef HAVE_CABS
#undef HAVE_CABSF
#undef HAVE_CABSL

/* These work for me in C++, even for the failing test case (2 ** 29),
   but the tests fail.  No idea if the functions I'm getting are used
   or not. */
#undef HAVE_LOG2
#undef HAVE_LOG2F
#undef HAVE_LOG2L

/* Complex square root does not use sign of zero to find branch
   cuts. */
#undef HAVE_CSQRT
#undef HAVE_CSQRTF
#undef HAVE_CSQRTL

/* C++ Real asinh works fine, complex asinh says asinh(1e-20+0j) =
   0 */
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL

/* _check_branch_cut(np.arcsin, ...) fails */
#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL

/* Branch cuts for arccosine also fail */
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL

/* Branch cuts for arctan fail as well */
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL

/* check_loss_of_precision fails in arctanh */
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL

/* modfl seems to be segfaulting at the moment */
#undef HAVE_MODFL

#endif

/* powl gives zero division warning on OS X, see gh-8307 */
#if defined(HAVE_POWL) && defined(NPY_OS_DARWIN)
#undef HAVE_POWL
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

#endif /* __GLIBC_PREREQ(2, 18) */
#endif /* defined(__GLIBC_PREREQ) */

#endif /* defined(HAVE_FEATURES_H) */

#endif

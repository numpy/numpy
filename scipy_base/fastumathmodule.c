#include "Python.h"
#include "Numeric/arrayobject.h"
#include "Numeric/ufuncobject.h"
#include "abstract.h"
#include <math.h>
#include "mconf_lite.h"

/* Fast umath module whose functions do not check for range and domain
   errors.

   Replacement for umath + additions for isnan, isfinite, and isinf
   Also allows comparison operations on complex numbers (just compares
   the real part) and logical operations.

   All logical operations return UBYTE arrays.
*/

#if defined _ISOC99_SOURCE || defined _XOPEN_SOURCE_EXTENDED \
    || defined _BSD_SOURCE || defined _SVID_SOURCE
#define HAVE_INVERSE_HYPERBOLIC 1
#endif

/* Wrapper to include the correct version */

#ifdef PyArray_UNSIGNED_TYPES
#include "fastumath_unsigned.inc"
#else
#include "fastumath_nounsigned.inc"
#endif

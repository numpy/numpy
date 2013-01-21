#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"
#include "numpy/numpyconfig.h"

/* Disable broken MS math functions */
#if defined(_MSC_VER) || defined(__MINGW32_VERSION)
#undef HAVE_ATAN2
#undef HAVE_HYPOT
#endif

/* Safe to use ldexp and frexp for long double for MSVC builds */
#if (NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE) || defined(_MSC_VER)
    #ifdef HAVE_LDEXP
        #define HAVE_LDEXPL 1
    #endif
    #ifdef HAVE_FREXP
        #define HAVE_FREXPL 1
    #endif
#endif

/* Disable broken Sun Workshop Pro math functions */
#ifdef __SUNPRO_C
#undef HAVE_ATAN2
#endif

#endif

#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"
#include "numpy/numpyconfig.h"

/* Disable broken MS math functions */
#if defined(_MSC_VER) || defined(__MINGW32_VERSION)
#undef HAVE_ATAN2
#undef HAVE_HYPOT
#endif

/*
 * largest alignment the copy loops might require
 * required as string, void and complex types might get copied using larger
 * instructions than required to operate on them. E.g. complex float is copied
 * in 8 byte moves but arithmetic on them only loads in 4 byte moves.
 * the sparc platform may need that alignment for long doubles.
 * amd64 is not harmed much by the bloat as the system provides 16 byte
 * alignment by default.
 */
#define NPY_MAX_COPY_ALIGNMENT 16

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

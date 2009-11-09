#ifndef _NPY_NPY_FPMATH_H_
#define _NPY_NPY_FPMATH_H_

#include "npy_os.h"
#include "npy_cpu.h"
#include "npy_common.h"

#ifdef NPY_OS_DARWIN
    /* This hardcoded logic is fragile, but universal builds makes it
     * difficult to detect arch-specific features */

    /* MAC OS X < 10.4 and gcc < 4 does not support proper long double, and
     * is the same as double on those platforms */
    #if NPY_BITSOF_LONGDOUBLE == NPY_BITSOF_DOUBLE
        /* This assumes that FPU and ALU have the same endianness */
        #if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
            #define NPY_HAVE_LDOUBLE_IEEE_DOUBLE_LE
        #elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN
            #define NPY_HAVE_LDOUBLE_IEEE_DOUBLE_BE
        #else
            #error Endianness undefined ?
        #endif
    #else
        #if defined(NPY_CPU_X86)
            #define NPY_HAVE_LDOUBLE_EXT_12_BYTES_LE
        #elif defined(NPY_CPU_AMD64)
            #define NPY_HAVE_LDOUBLE_EXT_16_BYTES_LE
        #elif defined(NPY_CPU_PPC) || defined(NPY_CPU_PPC64)
            #define NPY_HAVE_LDOUBLE_IEEE_QUAD_BE
        #endif
    #endif
#endif

#if !defined(NPY_HAVE_LDOUBLE_IEEE_QUAD_BE) || \
    !defined(NPY_HAVE_LDOUBLE_IEEE_QUAD_LE) || \
    !defined(NPY_HAVE_LDOUBLE_DOUBLE_LE) || \
    !defined(NPY_HAVE_LDOUBLE_DOUBLE_BE) || \
    !defined(NPY_HAVE_LDOUBLE_EXT_16_BYTES_LE) || \
    !defined(NPY_HAVE_LDOUBLE_EXT_12_BYTES_LE) ||
    #error No long double representation defined
#endif

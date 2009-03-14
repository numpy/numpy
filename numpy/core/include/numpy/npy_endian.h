#ifndef _NPY_ENDIAN_H_
#define _NPY_ENDIAN_H_

/*
 * NPY_BYTE_ORDER is set to the same value as BYTE_ORDER set by glibc in
 * endian.h
 */

#ifdef NPY_HAVE_ENDIAN_H
    /* Use endian.h if available */
    #include <endian.h>
    #define NPY_BYTE_ORDER __BYTE_ORDER
    #if (__BYTE_ORDER == __LITTLE_ENDIAN)
        #define NPY_LITTLE_ENDIAN
    #elif (__BYTE_ORDER == __BIG_ENDIAN)
        #define NPY_BIG_ENDIAN
    #else
        #error Unknown machine endianness detected.
    #endif
#else
    /* Set endianness info using target CPU */
    #include "npy_cpu.h"

    #if defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64)\
            || defined(NPY_CPU_IA64)
        #define NPY_LITTLE_ENDIAN
        #define NPY_BYTE_ORDER 1234
    #elif defined(NPY_CPU_PPC) || defined(NPY_CPU_SPARC)\
            || defined(NPY_CPU_S390) || defined(NPY_CPU_PARISC)
        #define NPY_BIG_ENDIAN
        #define NPY_BYTE_ORDER 4321
    #else
        #error Unknown CPU: can not set endianness
    #endif
#endif

#endif

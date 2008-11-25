#ifndef _NPY_ENDIAN_H_
#define _NPY_ENDIAN_H_

/* NPY_BYTE_ORDER is set to the same value as BYTE_ORDER set by glibc in
 * endian.h */

#ifdef NPY_HAVE_ENDIAN_H
        /* Use endian.h if available */
        #include <endian.h>
        #define NPY_BYTE_ODER __BYTE_ORDER
        #if (__BYTE_ORDER == __LITTLE_ENDIAN)
                #define NPY_LITTLE_ENDIAN
        #elif (__BYTE_ORDER == __BIG_ENDIAN)
                #define NPY_BYTE_ODER __BYTE_ORDER
        #else
                #error Unknown machine endianness detected.
        #endif
#else
        /* Set endianness info using target CPU */
        #include "cpuarch.h"
        
        #if defined(NPY_X86) || defined(NPY_AMD64)
                        #define NPY_LITTLE_ENDIAN
                        #define NPY_BYTE_ORDER 1234
        #elif defined(NPY_PPC) || defined(NPY_SPARC) || defined(NPY_S390) || \
              defined(NPY_PA_RISC)
                        #define NPY_BIG_ENDIAN
                        #define NPY_BYTE_ORDER 4321
        #endif
#endif

#endif

/*
 * This set (target) cpu specific macros:
 *      - NPY_TARGET_CPU: target CPU type. Possible values:
 *              NPY_X86
 *              NPY_AMD64
 *              NPY_PPC
 *              NPY_SPARC
 *              NPY_S390
 *              NPY_PA_RISC
 */
#ifndef _NPY_CPUARCH_H_
#define _NPY_CPUARCH_H_

#if defined ( _i386_ ) || defined( __i386__ )
        /* __i386__ is defined by gcc and Intel compiler on Linux, _i386_ by
        VS compiler */
        #define NPY_TARGET_CPU NPY_X86
#elif defined(__x86_64__) || defined(__amd64__)
        /* both __x86_64__ and __amd64__ are defined by gcc */
        #define NPY_TARGET_CPU NPY_AMD64
#elif defined(__ppc__) || defined(__powerpc__)
        /* __ppc__ is defined by gcc, I remember having seen __powerpc__ once,
         * but can't find it ATM */
        #define NPY_TARGET_CPU NPY_PPC
#elif defined(__sparc__) || defined(__sparc)
        /* __sparc__ is defined by gcc and Forte (e.g. Sun) compilers */
        #define NPY_TARGET_CPU NPY_SPARC
#elif defined(__s390__)
        #define NPY_TARGET_CPU NPY_S390
#elif defined(__parisc__)
        /* XXX: Not sure about this one... */
        #define NPY_TARGET_CPU NPY_PA_RISC
#else
        #error Unknown CPU, please report this to numpy maintainers with \
        information about your platform
#endif

#endif

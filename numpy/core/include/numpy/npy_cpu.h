/*
 * This set (target) cpu specific macros:
 *      - Possible values:
 *              NPY_CPU_X86
 *              NPY_CPU_AMD64
 *              NPY_CPU_PPC
 *              NPY_CPU_PPC64
 *              NPY_CPU_SPARC
 *              NPY_CPU_S390
 *              NPY_CPU_PARISC
 */
#ifndef _NPY_CPUARCH_H_
#define _NPY_CPUARCH_H_

#if defined ( _i386_ ) || defined( __i386__ )
        /* __i386__ is defined by gcc and Intel compiler on Linux, _i386_ by
        VS compiler */
        #define NPY_CPU_X86
#elif defined(__x86_64__) || defined(__amd64__)
        /* both __x86_64__ and __amd64__ are defined by gcc */
        #define NPY_CPU_AMD64
#elif defined(__ppc__) || defined(__powerpc__)
        /* __ppc__ is defined by gcc, I remember having seen __powerpc__ once,
         * but can't find it ATM */
        #define NPY_CPU_PPC
#elif defined(__ppc64__)
        #define NPY_CPU_PPC64
#elif defined(__sparc__) || defined(__sparc)
        /* __sparc__ is defined by gcc and Forte (e.g. Sun) compilers */
        #define NPY_CPU_SPARC
#elif defined(__s390__)
        #define NPY_CPU_S390
#elif defined(__parisc__)
        /* XXX: Not sure about this one... */
        #define NPY_CPU_PARISC
#else
        #error Unknown CPU, please report this to numpy maintainers with \
        information about your platform (OS, CPU and compiler)
#endif

#endif

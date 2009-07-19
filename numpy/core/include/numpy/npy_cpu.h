/*
 * This set (target) cpu specific macros:
 *      - Possible values:
 *              NPY_CPU_X86
 *              NPY_CPU_AMD64
 *              NPY_CPU_PPC
 *              NPY_CPU_PPC64
 *              NPY_CPU_SPARC
 *              NPY_CPU_S390
 *              NPY_CPU_IA64
 *              NPY_CPU_HPPA
 *              NPY_CPU_ALPHA
 */
#ifndef _NPY_CPUARCH_H_
#define _NPY_CPUARCH_H_

#if defined( __i386__ ) || defined(i386) || defined(_M_IX86)
    /*
     * __i386__ is defined by gcc and Intel compiler on Linux,
     * _M_IX86 by VS compiler,
     * i386 by Sun compilers on opensolaris at least
     */
    #define NPY_CPU_X86
#elif defined(__x86_64__) || defined(__amd64__) || defined(__x86_64) || defined(_M_AMD64)
    /*
     * both __x86_64__ and __amd64__ are defined by gcc
     * __x86_64 defined by sun compiler on opensolaris at least
     * _M_AMD64 defined by MS compiler
     */
    #define NPY_CPU_AMD64
#elif defined(__ppc__) || defined(__powerpc__)
    /*
     * __ppc__ is defined by gcc, I remember having seen __powerpc__ once,
     * but can't find it ATM
     */
    #define NPY_CPU_PPC
#elif defined(__ppc64__)
    #define NPY_CPU_PPC64
#elif defined(__sparc__) || defined(__sparc)
    /* __sparc__ is defined by gcc and Forte (e.g. Sun) compilers */
    #define NPY_CPU_SPARC
#elif defined(__s390__)
    #define NPY_CPU_S390
#elif defined(__ia64)
    #define NPY_CPU_IA64
#elif defined(__hppa__)
    #define NPY_CPU_HPPA
#elif defined(__alpha__)
    #define NPY_CPU_ALPHA
#else
    #error Unknown CPU, please report this to numpy maintainers with \
    information about your platform (OS, CPU and compiler)
#endif

#endif

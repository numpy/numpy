#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include "npy_config.h"

#include "cpuid.h"

#define XCR_XFEATURE_ENABLED_MASK 0x0
#define XSTATE_SSE 0x2
#define XSTATE_YMM 0x4
#define XSTATE_ZMM 0x70

/*
 * verify the OS supports avx instructions
 * it can be disabled in some OS, e.g. with the nosavex boot option of linux
 */
static NPY_INLINE
int os_avx_support(void)
{
#if HAVE_XGETBV
    /*
     * use bytes for xgetbv to avoid issues with compiler not knowing the
     * instruction
     */
    unsigned int eax, edx;
    unsigned int ecx = XCR_XFEATURE_ENABLED_MASK;
    __asm__("xgetbv" : "=a" (eax), "=d" (edx) : "c" (ecx));
    return (eax & (XSTATE_SSE | XSTATE_YMM)) == (XSTATE_SSE | XSTATE_YMM);
#else
    return 0;
#endif
}

static NPY_INLINE
int os_avx512_support(void)
{
#if HAVE_XGETBV
    unsigned int eax, edx;
    unsigned int ecx = XCR_XFEATURE_ENABLED_MASK;
    unsigned int xcr0 = XSTATE_ZMM | XSTATE_YMM | XSTATE_SSE;
    __asm__("xgetbv" : "=a" (eax), "=d" (edx) : "c" (ecx));
    return (eax & xcr0) == xcr0;
#else
    return 0;
#endif
}

static NPY_INLINE
int cpu_supports_fma(void)
{
#ifdef __x86_64__
    unsigned int feature = 0x01;
    unsigned int a, b, c, d;
    __asm__ volatile (
        "cpuid"				"\n\t"
	: "=a" (a), "=b" (b), "=c" (c), "=d" (d)
	: "a" (feature));
    /*
     * FMA is the 12th bit of ECX
     */
    return (c >> 12) & 1;
#else
    return 0;
#endif
}

/*
 * Primitive cpu feature detect function
 * Currently only supports checking for avx on gcc compatible compilers.
 */
NPY_NO_EXPORT int
npy_cpu_supports(const char * feature)
{
#ifdef HAVE___BUILTIN_CPU_SUPPORTS
    if (strcmp(feature, "avx512f") == 0) {
#ifdef HAVE___BUILTIN_CPU_SUPPORTS_AVX512F
        return __builtin_cpu_supports("avx512f") && os_avx512_support();
#else
        return 0;
#endif
    }
    else if (strcmp(feature, "fma") == 0) {
        return cpu_supports_fma() && __builtin_cpu_supports("avx2") && os_avx_support();
    }
    else if (strcmp(feature, "avx2") == 0) {
        return __builtin_cpu_supports("avx2") && os_avx_support();
    }
    else if (strcmp(feature, "avx") == 0) {
        return __builtin_cpu_supports("avx") && os_avx_support();
    }
#endif

    return 0;
}

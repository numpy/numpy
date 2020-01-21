#ifndef _NPY_CPU_FEATURES_H_
#define _NPY_CPU_FEATURES_H_

#include "numpy/numpyconfig.h" // for NPY_VISIBILITY_HIDDEN
#include <Python.h> // for PyObject

#ifdef __cplusplus
extern "C" {
#endif

enum npy_cpu_features
{
    NPY_CPU_FEATURE_NONE = 0,
    // X86
    NPY_CPU_FEATURE_MMX               = 1,
    NPY_CPU_FEATURE_SSE               = 2,
    NPY_CPU_FEATURE_SSE2              = 3,
    NPY_CPU_FEATURE_SSE3              = 4,
    NPY_CPU_FEATURE_SSSE3             = 5,
    NPY_CPU_FEATURE_SSE41             = 6,
    NPY_CPU_FEATURE_POPCNT            = 7,
    NPY_CPU_FEATURE_SSE42             = 8,
    NPY_CPU_FEATURE_AVX               = 9,
    NPY_CPU_FEATURE_F16C              = 10,
    NPY_CPU_FEATURE_XOP               = 11,
    NPY_CPU_FEATURE_FMA4              = 12,
    NPY_CPU_FEATURE_FMA3              = 13,
    NPY_CPU_FEATURE_AVX2              = 14,
    NPY_CPU_FEATURE_FMA               = 15, // AVX2 & FMA3, provides backward compatibility

    NPY_CPU_FEATURE_AVX512F           = 30,
    NPY_CPU_FEATURE_AVX512CD          = 31,
    NPY_CPU_FEATURE_AVX512ER          = 32,
    NPY_CPU_FEATURE_AVX512PF          = 33,
    NPY_CPU_FEATURE_AVX5124FMAPS      = 34,
    NPY_CPU_FEATURE_AVX5124VNNIW      = 35,
    NPY_CPU_FEATURE_AVX512VPOPCNTDQ   = 36,
    NPY_CPU_FEATURE_AVX512BW          = 37,
    NPY_CPU_FEATURE_AVX512DQ          = 38,
    NPY_CPU_FEATURE_AVX512VL          = 39,
    NPY_CPU_FEATURE_AVX512IFMA        = 40,
    NPY_CPU_FEATURE_AVX512VBMI        = 41,
    NPY_CPU_FEATURE_AVX512VNNI        = 42,
    NPY_CPU_FEATURE_AVX512VBMI2       = 43,
    NPY_CPU_FEATURE_AVX512BITALG      = 44,

    // X86 CPU Groups
    // Knights Landing (F,CD,ER,PF)
    NPY_CPU_FEATURE_AVX512_KNL        = 101,
    // Knights Mill    (F,CD,ER,PF,4FMAPS,4VNNIW,VPOPCNTDQ)
    NPY_CPU_FEATURE_AVX512_KNM        = 102,
    // Skylake-X       (F,CD,BW,DQ,VL)
    NPY_CPU_FEATURE_AVX512_SKX        = 103,
    // Cascade Lake    (F,CD,BW,DQ,VL,VNNI)
    NPY_CPU_FEATURE_AVX512_CLX        = 104,
    // Cannon Lake     (F,CD,BW,DQ,VL,IFMA,VBMI)
    NPY_CPU_FEATURE_AVX512_CNL        = 105,
    // Ice Lake        (F,CD,BW,DQ,VL,IFMA,VBMI,VNNI,VBMI2,BITALG,VPOPCNTDQ)
    NPY_CPU_FEATURE_AVX512_ICL        = 106,

    // IBM/POWER VSX
    // POWER7
    NPY_CPU_FEATURE_VSX               = 200,
    // POWER8
    NPY_CPU_FEATURE_VSX2              = 201,
    // POWER9
    NPY_CPU_FEATURE_VSX3              = 202,

    // ARM
    NPY_CPU_FEATURE_NEON              = 300,
    NPY_CPU_FEATURE_NEON_FP16         = 301,
    // FMA
    NPY_CPU_FEATURE_NEON_VFPV4        = 302,
    // Advanced SIMD
    NPY_CPU_FEATURE_ASIMD             = 303,
    // ARMv8.2 half-precision
    NPY_CPU_FEATURE_FPHP              = 304,
    // ARMv8.2 half-precision vector arithm
    NPY_CPU_FEATURE_ASIMDHP           = 305,
    // ARMv8.2 dot product
    NPY_CPU_FEATURE_ASIMDDP           = 306,
    // ARMv8.2 single&half-precision multiply
    NPY_CPU_FEATURE_ASIMDFHM          = 307,

    NPY_CPU_FEATURE_MAX
};

/*
 * Initialize CPU features
 * return 0 on success otherwise return -1
*/
NPY_VISIBILITY_HIDDEN int
npy_cpu_init(void);

/*
 * return 0 if CPU feature isn't available
 * note: `npy_cpu_init` must be called first otherwise it will always return 0
*/
NPY_VISIBILITY_HIDDEN int
npy_cpu_have(int feature_id);

#define NPY_CPU_HAVE(FEATURE_NAME) \
npy_cpu_have(NPY_CPU_FEATURE_##FEATURE_NAME)

/*
 * return a new dictionary contains CPU feature names
 * with runtime availability.
 * same as npy_cpu_have, `npy_cpu_init` must be called first.
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_features_dict(void);

#ifdef __cplusplus
}
#endif

#endif // _NPY_CPU_FEATURES_H_

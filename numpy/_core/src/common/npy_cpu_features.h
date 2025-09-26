#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_FEATURES_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPU_FEATURES_H_

#include <Python.h> // for PyObject
#include "numpy/numpyconfig.h" // for NPY_VISIBILITY_HIDDEN

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
    NPY_CPU_FEATURE_LAHF              = 15,
    NPY_CPU_FEATURE_CX16              = 16,
    NPY_CPU_FEATURE_MOVBE             = 17,
    NPY_CPU_FEATURE_BMI               = 18,
    NPY_CPU_FEATURE_BMI2              = 19,
    NPY_CPU_FEATURE_LZCNT             = 20,
    NPY_CPU_FEATURE_GFNI              = 21,
    NPY_CPU_FEATURE_VAES              = 22,
    NPY_CPU_FEATURE_VPCLMULQDQ        = 23,
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
    NPY_CPU_FEATURE_AVX512FP16        = 45,
    NPY_CPU_FEATURE_AVX512BF16        = 46,


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
    // Ice Lake        (F,CD,BW,DQ,VL,IFMA,VBMI,VNNI,VBMI2,BITALG,VPOPCNTDQ,GFNI,VPCLMULDQ,VAES)
    NPY_CPU_FEATURE_AVX512_ICL        = 106,
    // Sapphire Rapids (Ice Lake, AVX512FP16, AVX512BF16)
    NPY_CPU_FEATURE_AVX512_SPR        = 107,
    // x86-64-v2 microarchitectures (SSE[1-4.*], POPCNT, LAHF, CX16)
    // On 32-bit, cx16 is not available so it is not included
    NPY_CPU_FEATURE_X86_V2 = 108,
    // x86-64-v3 microarchitectures (X86_V2, AVX, AVX2, FMA3, BMI, BMI2, LZCNT, F16C, MOVBE)
    NPY_CPU_FEATURE_X86_V3 = 109,
    // x86-64-v4 microarchitectures (X86_V3, AVX512F, AVX512CD, AVX512VL, AVX512BW, AVX512DQ)
    NPY_CPU_FEATURE_X86_V4 = NPY_CPU_FEATURE_AVX512_SKX,

    // IBM/POWER VSX
    // POWER7
    NPY_CPU_FEATURE_VSX               = 200,
    // POWER8
    NPY_CPU_FEATURE_VSX2              = 201,
    // POWER9
    NPY_CPU_FEATURE_VSX3              = 202,
    // POWER10
    NPY_CPU_FEATURE_VSX4              = 203,

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
    // Scalable Vector Extensions (SVE)
    NPY_CPU_FEATURE_SVE               = 308,

    // IBM/ZARCH
    NPY_CPU_FEATURE_VX                = 350,

    // Vector-Enhancements Facility 1
    NPY_CPU_FEATURE_VXE               = 351,

    // Vector-Enhancements Facility 2
    NPY_CPU_FEATURE_VXE2              = 352,

    // RISC-V
    NPY_CPU_FEATURE_RVV               = 400,

    // LOONGARCH
    NPY_CPU_FEATURE_LSX               = 500,

    NPY_CPU_FEATURE_MAX
};

/*
 * Initialize CPU features
 *
 * This function
 *  - detects runtime CPU features
 *  - check that baseline CPU features are present
 *  - uses 'NPY_DISABLE_CPU_FEATURES' to disable dispatchable features
 *  - uses 'NPY_ENABLE_CPU_FEATURES' to enable dispatchable features
 *
 * It will set a RuntimeError when
 *  - CPU baseline features from the build are not supported at runtime
 *  - 'NPY_DISABLE_CPU_FEATURES' tries to disable a baseline feature
 *  - 'NPY_DISABLE_CPU_FEATURES' and 'NPY_ENABLE_CPU_FEATURES' are
 *    simultaneously set
 *  - 'NPY_ENABLE_CPU_FEATURES' tries to enable a feature that is not supported
 *    by the machine or build
 *  - 'NPY_ENABLE_CPU_FEATURES' tries to enable a feature when the project was
 *    not built with any feature optimization support
 *
 * It will set an ImportWarning when:
 *  - 'NPY_DISABLE_CPU_FEATURES' tries to disable a feature that is not supported
 *    by the machine or build
 *  - 'NPY_DISABLE_CPU_FEATURES' or 'NPY_ENABLE_CPU_FEATURES' tries to
 *    disable/enable a feature when the project was not built with any feature
 *    optimization support
 *
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
/*
 * Return a new a Python list contains the minimal set of required optimizations
 * that supported by the compiler and platform according to the specified
 * values to command argument '--cpu-baseline'.
 *
 * This function is mainly used to implement umath's attribute '__cpu_baseline__',
 * and the items are sorted from the lowest to highest interest.
 *
 * For example, according to the default build configuration and by assuming the compiler
 * support all the involved optimizations then the returned list should equivalent to:
 *
 * On x86: ['SSE', 'SSE2']
 * On x64: ['SSE', 'SSE2', 'SSE3']
 * On armhf: []
 * On aarch64: ['NEON', 'NEON_FP16', 'NEON_VPFV4', 'ASIMD']
 * On ppc64: []
 * On ppc64le: ['VSX', 'VSX2']
 * On s390x: []
 * On any other arch or if the optimization is disabled: []
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_baseline_list(void);
/*
 * Return a new a Python list contains the dispatched set of additional optimizations
 * that supported by the compiler and platform according to the specified
 * values to command argument '--cpu-dispatch'.
 *
 * This function is mainly used to implement umath's attribute '__cpu_dispatch__',
 * and the items are sorted from the lowest to highest interest.
 *
 * For example, according to the default build configuration and by assuming the compiler
 * support all the involved optimizations then the returned list should equivalent to:
 *
 * On x86: ['SSE3', 'SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', ...]
 * On x64: ['SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', ...]
 * On armhf: ['NEON', 'NEON_FP16', 'NEON_VPFV4', 'ASIMD', 'ASIMDHP', 'ASIMDDP', 'ASIMDFHM']
 * On aarch64: ['ASIMDHP', 'ASIMDDP', 'ASIMDFHM']
 * On ppc64:  ['VSX', 'VSX2', 'VSX3', 'VSX4']
 * On ppc64le: ['VSX3', 'VSX4']
 * On s390x: ['VX', 'VXE', VXE2]
 * On any other arch or if the optimization is disabled: []
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_dispatch_list(void);

#ifdef __cplusplus
}
#endif

#endif  // NUMPY_CORE_SRC_COMMON_NPY_CPU_FEATURES_H_

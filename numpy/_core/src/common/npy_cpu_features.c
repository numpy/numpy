#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h" // To guarantee the CPU baseline definitions are in scope.
#include "numpy/npy_common.h"
#include "numpy/npy_cpu.h" // To guarantee the CPU definitions are in scope.

/******************** Private Definitions *********************/

// This is initialized during module initialization and thereafter immutable.
// We don't include it in the global data struct because the definitions in
// this file are shared by the _simd, _umath_tests, and
// _multiarray_umath modules

// Hold all CPU features boolean values
static unsigned char npy__cpu_have[NPY_CPU_FEATURE_MAX];

/******************** Private Declarations *********************/

// Almost detect all CPU features in runtime
static void
npy__cpu_init_features(void);
/*
 * Enable or disable CPU dispatched features at runtime if the environment variable
 * `NPY_ENABLE_CPU_FEATURES`  or  `NPY_DISABLE_CPU_FEATURES`
 * depends on the value of boolean parameter `disable`(toggle).
 *
 * Multiple features can be present, and separated by space, comma, or tab.
 * Raises an error if parsing fails or if the feature was not enabled or disabled
*/
static int
npy__cpu_check_env(int disable, const char *env);

/* Ensure the build's CPU baseline features are supported at runtime */
static int
npy__cpu_validate_baseline(void);

/******************** Public Definitions *********************/

NPY_VISIBILITY_HIDDEN int
npy_cpu_have(int feature_id)
{
    if (feature_id <= NPY_CPU_FEATURE_NONE || feature_id >= NPY_CPU_FEATURE_MAX)
        return 0;
    return npy__cpu_have[feature_id];
}

NPY_VISIBILITY_HIDDEN int
npy_cpu_init(void)
{
    npy__cpu_init_features();
    if (npy__cpu_validate_baseline() < 0) {
        return -1;
    }
    char *enable_env = getenv("NPY_ENABLE_CPU_FEATURES");
    char *disable_env = getenv("NPY_DISABLE_CPU_FEATURES");
    int is_enable = enable_env && enable_env[0];
    int is_disable = disable_env && disable_env[0];
    if (is_enable & is_disable) {
        PyErr_Format(PyExc_ImportError,
            "Both NPY_DISABLE_CPU_FEATURES and NPY_ENABLE_CPU_FEATURES "
            "environment variables cannot be set simultaneously."
        );
        return -1;
    }
    if (is_enable | is_disable) {
        if (npy__cpu_check_env(is_disable, is_disable ? disable_env : enable_env) < 0) {
            return -1;
        }
    }
    return 0;
}

static struct {
  enum npy_cpu_features feature;
  char const *string;
} features[] = {{NPY_CPU_FEATURE_MMX, "MMX"},
                {NPY_CPU_FEATURE_SSE, "SSE"},
                {NPY_CPU_FEATURE_SSE2, "SSE2"},
                {NPY_CPU_FEATURE_SSE3, "SSE3"},
                {NPY_CPU_FEATURE_SSSE3, "SSSE3"},
                {NPY_CPU_FEATURE_SSE41, "SSE41"},
                {NPY_CPU_FEATURE_POPCNT, "POPCNT"},
                {NPY_CPU_FEATURE_SSE42, "SSE42"},
                {NPY_CPU_FEATURE_AVX, "AVX"},
                {NPY_CPU_FEATURE_F16C, "F16C"},
                {NPY_CPU_FEATURE_XOP, "XOP"},
                {NPY_CPU_FEATURE_FMA4, "FMA4"},
                {NPY_CPU_FEATURE_FMA3, "FMA3"},
                {NPY_CPU_FEATURE_AVX2, "AVX2"},
                {NPY_CPU_FEATURE_AVX512F, "AVX512F"},
                {NPY_CPU_FEATURE_AVX512CD, "AVX512CD"},
                {NPY_CPU_FEATURE_AVX512ER, "AVX512ER"},
                {NPY_CPU_FEATURE_AVX512PF, "AVX512PF"},
                {NPY_CPU_FEATURE_AVX5124FMAPS, "AVX5124FMAPS"},
                {NPY_CPU_FEATURE_AVX5124VNNIW, "AVX5124VNNIW"},
                {NPY_CPU_FEATURE_AVX512VPOPCNTDQ, "AVX512VPOPCNTDQ"},
                {NPY_CPU_FEATURE_AVX512VL, "AVX512VL"},
                {NPY_CPU_FEATURE_AVX512BW, "AVX512BW"},
                {NPY_CPU_FEATURE_AVX512DQ, "AVX512DQ"},
                {NPY_CPU_FEATURE_AVX512VNNI, "AVX512VNNI"},
                {NPY_CPU_FEATURE_AVX512IFMA, "AVX512IFMA"},
                {NPY_CPU_FEATURE_AVX512VBMI, "AVX512VBMI"},
                {NPY_CPU_FEATURE_AVX512VBMI2, "AVX512VBMI2"},
                {NPY_CPU_FEATURE_AVX512BITALG, "AVX512BITALG"},
                {NPY_CPU_FEATURE_AVX512FP16 , "AVX512FP16"},
                {NPY_CPU_FEATURE_AVX512_KNL, "AVX512_KNL"},
                {NPY_CPU_FEATURE_AVX512_KNM, "AVX512_KNM"},
                {NPY_CPU_FEATURE_AVX512_SKX, "AVX512_SKX"},
                {NPY_CPU_FEATURE_AVX512_CLX, "AVX512_CLX"},
                {NPY_CPU_FEATURE_AVX512_CNL, "AVX512_CNL"},
                {NPY_CPU_FEATURE_AVX512_ICL, "AVX512_ICL"},
                {NPY_CPU_FEATURE_AVX512_SPR, "AVX512_SPR"},
                {NPY_CPU_FEATURE_VSX, "VSX"},
                {NPY_CPU_FEATURE_VSX2, "VSX2"},
                {NPY_CPU_FEATURE_VSX3, "VSX3"},
                {NPY_CPU_FEATURE_VSX4, "VSX4"},
                {NPY_CPU_FEATURE_VX, "VX"},
                {NPY_CPU_FEATURE_VXE, "VXE"},
                {NPY_CPU_FEATURE_VXE2, "VXE2"},
                {NPY_CPU_FEATURE_NEON, "NEON"},
                {NPY_CPU_FEATURE_NEON_FP16, "NEON_FP16"},
                {NPY_CPU_FEATURE_NEON_VFPV4, "NEON_VFPV4"},
                {NPY_CPU_FEATURE_ASIMD, "ASIMD"},
                {NPY_CPU_FEATURE_FPHP, "FPHP"},
                {NPY_CPU_FEATURE_ASIMDHP, "ASIMDHP"},
                {NPY_CPU_FEATURE_ASIMDDP, "ASIMDDP"},
                {NPY_CPU_FEATURE_ASIMDFHM, "ASIMDFHM"},
                {NPY_CPU_FEATURE_SVE, "SVE"},
                {NPY_CPU_FEATURE_RVV, "RVV"},
                {NPY_CPU_FEATURE_LSX, "LSX"}};


NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_features_dict(void)
{
    PyObject *dict = PyDict_New();
    if (dict) {
        for(unsigned i = 0; i < sizeof(features)/sizeof(features[0]); ++i)
            if (PyDict_SetItemString(dict, features[i].string,
                npy__cpu_have[features[i].feature] ? Py_True : Py_False) < 0) {
                Py_DECREF(dict);
                return NULL;
            }
    }
    return dict;
}

#define NPY__CPU_PYLIST_APPEND_CB(FEATURE, LIST) \
    item = PyUnicode_FromString(NPY_TOSTRING(FEATURE)); \
    if (item == NULL) { \
        Py_DECREF(LIST); \
        return NULL; \
    } \
    PyList_SET_ITEM(LIST, index++, item);

NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_baseline_list(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    PyObject *list = PyList_New(NPY_WITH_CPU_BASELINE_N), *item;
    int index = 0;
    if (list != NULL) {
        NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_PYLIST_APPEND_CB, list)
    }
    return list;
#else
    return PyList_New(0);
#endif
}

NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_dispatch_list(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    PyObject *list = PyList_New(NPY_WITH_CPU_DISPATCH_N), *item;
    int index = 0;
    if (list != NULL) {
        NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_PYLIST_APPEND_CB, list)
    }
    return list;
#else
    return PyList_New(0);
#endif
}

/******************** Private Definitions *********************/
#define NPY__CPU_FEATURE_ID_CB(FEATURE, WITH_FEATURE)     \
    if (strcmp(NPY_TOSTRING(FEATURE), WITH_FEATURE) == 0) \
        return NPY_CAT(NPY_CPU_FEATURE_, FEATURE);
/**
 * Returns CPU feature's ID, if the 'feature' was part of baseline
 * features that had been configured via --cpu-baseline
 * otherwise it returns 0
*/
static inline int
npy__cpu_baseline_fid(const char *feature)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_FEATURE_ID_CB, feature)
#endif
    return 0;
}
/**
 * Returns CPU feature's ID, if the 'feature' was part of dispatched
 * features that had been configured via --cpu-dispatch
 * otherwise it returns 0
*/
static inline int
npy__cpu_dispatch_fid(const char *feature)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_FEATURE_ID_CB, feature)
#endif
    return 0;
}

static int
npy__cpu_validate_baseline(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    char baseline_failure[sizeof(NPY_WITH_CPU_BASELINE) + 1];
    char *fptr = &baseline_failure[0];

    #define NPY__CPU_VALIDATE_CB(FEATURE, DUMMY)                  \
        if (!npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)]) { \
            const int size = sizeof(NPY_TOSTRING(FEATURE));       \
            memcpy(fptr, NPY_TOSTRING(FEATURE), size);            \
            fptr[size] = ' '; fptr += size + 1;                   \
        }
    NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_VALIDATE_CB, DUMMY) // extra arg for msvc
    *fptr = '\0';

    if (baseline_failure[0] != '\0') {
        *(fptr-1) = '\0'; // trim the last space
        PyErr_Format(PyExc_RuntimeError,
            "NumPy was built with baseline optimizations: \n"
            "(" NPY_WITH_CPU_BASELINE ") but your machine "
            "doesn't support:\n(%s).",
            baseline_failure
        );
        return -1;
    }
#endif
    return 0;
}

static int
npy__cpu_check_env(int disable, const char *env) {

    static const char *names[] = {
        "enable", "disable",
        "NPY_ENABLE_CPU_FEATURES", "NPY_DISABLE_CPU_FEATURES",
        "During parsing environment variable: 'NPY_ENABLE_CPU_FEATURES':\n",
        "During parsing environment variable: 'NPY_DISABLE_CPU_FEATURES':\n"
    };
    disable = disable ? 1 : 0;
    const char *act_name = names[disable];
    const char *env_name = names[disable + 2];
    const char *err_head = names[disable + 4];

#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    #define NPY__MAX_VAR_LEN 1024 // More than enough for this era
    size_t var_len = strlen(env) + 1;
    if (var_len > NPY__MAX_VAR_LEN) {
        PyErr_Format(PyExc_RuntimeError,
            "Length of environment variable '%s' is %d, only %d accepted",
            env_name, var_len, NPY__MAX_VAR_LEN
        );
        return -1;
    }
    char features[NPY__MAX_VAR_LEN];
    memcpy(features, env, var_len);

    char nexist[NPY__MAX_VAR_LEN];
    char *nexist_cur = &nexist[0];

    char notsupp[sizeof(NPY_WITH_CPU_DISPATCH) + 1];
    char *notsupp_cur = &notsupp[0];

    //comma and space including (htab, vtab, CR, LF, FF)
    const char *delim = ", \t\v\r\n\f";
    char *feature = strtok(features, delim);
    while (feature) {
        if (npy__cpu_baseline_fid(feature) > 0){
            if (disable) {
                PyErr_Format(PyExc_RuntimeError,
                    "%s"
                    "You cannot disable CPU feature '%s', since it is part of "
                    "the baseline optimizations:\n"
                    "(" NPY_WITH_CPU_BASELINE ").",
                    err_head, feature
                );
                return -1;
            } goto next;
        }
        // check if the feature is part of dispatched features
        int feature_id = npy__cpu_dispatch_fid(feature);
        if (feature_id == 0) {
            int flen = strlen(feature);
            memcpy(nexist_cur, feature, flen);
            nexist_cur[flen] = ' '; nexist_cur += flen + 1;
            goto next;
        }
        // check if the feature supported by the running machine
        if (!npy__cpu_have[feature_id]) {
            int flen = strlen(feature);
            memcpy(notsupp_cur, feature, flen);
            notsupp_cur[flen] = ' '; notsupp_cur += flen + 1;
            goto next;
        }
        // Finally we can disable or mark for enabling
        npy__cpu_have[feature_id] = disable ? 0:2;
    next:
        feature = strtok(NULL, delim);
    }
    if (!disable){
        // Disables any unmarked dispatched feature.
        #define NPY__CPU_DISABLE_DISPATCH_CB(FEATURE, DUMMY) \
            if(npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)] != 0)\
            {npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)]--;}\

        NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_DISABLE_DISPATCH_CB, DUMMY) // extra arg for msvc
    }

    *nexist_cur = '\0';
    if (nexist[0] != '\0') {
        *(nexist_cur-1) = '\0'; // trim the last space
        if (PyErr_WarnFormat(PyExc_ImportWarning, 1,
            "%sYou cannot %s CPU features (%s), since "
            "they are not part of the dispatched optimizations\n"
            "(" NPY_WITH_CPU_DISPATCH ").",
            err_head, act_name, nexist
        ) < 0) {
            return -1;
        }
    }

    #define NOTSUPP_BODY \
                "%s" \
                "You cannot %s CPU features (%s), since " \
                "they are not supported by your machine.", \
                err_head, act_name, notsupp

    *notsupp_cur = '\0';
    if (notsupp[0] != '\0') {
        *(notsupp_cur-1) = '\0'; // trim the last space
        if (!disable){
            PyErr_Format(PyExc_RuntimeError, NOTSUPP_BODY);
            return -1;
        }
    }
#else
    if (PyErr_WarnFormat(PyExc_ImportWarning, 1,
            "%s"
            "You cannot use environment variable '%s', since "
        #ifdef NPY_DISABLE_OPTIMIZATION
            "the NumPy library was compiled with optimization disabled.",
        #else
            "the NumPy library was compiled without any dispatched optimizations.",
        #endif
        err_head, env_name, act_name
    ) < 0) {
        return -1;
    }
#endif
    return 0;
}

/****************************************************************
 * This section is reserved to defining @npy__cpu_init_features
 * for each CPU architecture, please try to keep it clean. Ty
 ****************************************************************/

/***************** X86 ******************/

#if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)

#ifdef _MSC_VER
    #include <intrin.h>
#elif defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif

static int
npy__cpu_getxcr0(void)
{
#if defined(_MSC_VER) || defined (__INTEL_COMPILER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    /* named form of xgetbv not supported on OSX, so must use byte form, see:
     * https://github.com/asmjit/asmjit/issues/78
    */
    unsigned int eax, edx;
    __asm(".byte 0x0F, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return eax;
#else
    return 0;
#endif
}

static void
npy__cpu_cpuid(int reg[4], int func_id)
{
#if defined(_MSC_VER)
    __cpuidex(reg, func_id, 0);
#elif defined(__INTEL_COMPILER)
    __cpuid(reg, func_id);
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(NPY_CPU_X86) && defined(__PIC__)
        // %ebx may be the PIC register
        __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                "cpuid\n\t"
                "xchg{l}\t{%%}ebx, %1\n\t"
                : "=a" (reg[0]), "=r" (reg[1]), "=c" (reg[2]),
                  "=d" (reg[3])
                : "a" (func_id), "c" (0)
        );
    #else
        __asm__("cpuid\n\t"
                : "=a" (reg[0]), "=b" (reg[1]), "=c" (reg[2]),
                  "=d" (reg[3])
                : "a" (func_id), "c" (0)
        );
    #endif
#else
    reg[0] = 0;
#endif
}

static void
npy__cpu_init_features(void)
{
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

    // validate platform support
    int reg[] = {0, 0, 0, 0};
    npy__cpu_cpuid(reg, 0);
    if (reg[0] == 0) {
       npy__cpu_have[NPY_CPU_FEATURE_MMX]  = 1;
       npy__cpu_have[NPY_CPU_FEATURE_SSE]  = 1;
       npy__cpu_have[NPY_CPU_FEATURE_SSE2] = 1;
       #ifdef NPY_CPU_AMD64
           npy__cpu_have[NPY_CPU_FEATURE_SSE3] = 1;
       #endif
       return;
    }

    npy__cpu_cpuid(reg, 1);
    npy__cpu_have[NPY_CPU_FEATURE_MMX]    = (reg[3] & (1 << 23)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE]    = (reg[3] & (1 << 25)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE2]   = (reg[3] & (1 << 26)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE3]   = (reg[2] & (1 << 0))  != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSSE3]  = (reg[2] & (1 << 9))  != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE41]  = (reg[2] & (1 << 19)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_POPCNT] = (reg[2] & (1 << 23)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE42]  = (reg[2] & (1 << 20)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_F16C]   = (reg[2] & (1 << 29)) != 0;

    // check OSXSAVE
    if ((reg[2] & (1 << 27)) == 0)
        return;
    // check AVX OS support
    int xcr = npy__cpu_getxcr0();
    if ((xcr & 6) != 6)
        return;
    npy__cpu_have[NPY_CPU_FEATURE_AVX]    = (reg[2] & (1 << 28)) != 0;
    if (!npy__cpu_have[NPY_CPU_FEATURE_AVX])
        return;
    npy__cpu_have[NPY_CPU_FEATURE_FMA3]   = (reg[2] & (1 << 12)) != 0;

    // second call to the cpuid to get extended AMD feature bits
    npy__cpu_cpuid(reg, 0x80000001);
    npy__cpu_have[NPY_CPU_FEATURE_XOP]    = (reg[2] & (1 << 11)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_FMA4]   = (reg[2] & (1 << 16)) != 0;

    // third call to the cpuid to get extended AVX2 & AVX512 feature bits
    npy__cpu_cpuid(reg, 7);
    npy__cpu_have[NPY_CPU_FEATURE_AVX2]   = (reg[1] & (1 << 5))  != 0;
    npy__cpu_have[NPY_CPU_FEATURE_AVX2]   = npy__cpu_have[NPY_CPU_FEATURE_AVX2] &&
                                            npy__cpu_have[NPY_CPU_FEATURE_FMA3];
    if (!npy__cpu_have[NPY_CPU_FEATURE_AVX2])
        return;
    // detect AVX2 & FMA3
    npy__cpu_have[NPY_CPU_FEATURE_FMA]    = npy__cpu_have[NPY_CPU_FEATURE_FMA3];

    // check AVX512 OS support
    int avx512_os = (xcr & 0xe6) == 0xe6;
#if defined(__APPLE__) && defined(__x86_64__)
    /**
     * On darwin, machines with AVX512 support, by default, threads are created with
     * AVX512 masked off in XCR0 and an AVX-sized savearea is used.
     * However, AVX512 capabilities are advertised in the commpage and via sysctl.
     * for more information, check:
     *  - https://github.com/apple/darwin-xnu/blob/0a798f6738bc1db01281fc08ae024145e84df927/osfmk/i386/fpu.c#L175-L201
     *  - https://github.com/golang/go/issues/43089
     *  - https://github.com/numpy/numpy/issues/19319
     */
    if (!avx512_os) {
        npy_uintp commpage64_addr = 0x00007fffffe00000ULL;
        npy_uint16 commpage64_ver = *((npy_uint16*)(commpage64_addr + 0x01E));
        // cpu_capabilities64 undefined in versions < 13
        if (commpage64_ver > 12) {
            npy_uint64 commpage64_cap = *((npy_uint64*)(commpage64_addr + 0x010));
            avx512_os = (commpage64_cap & 0x0000004000000000ULL) != 0;
        }
    }
#endif
    if (!avx512_os) {
        return;
    }
    npy__cpu_have[NPY_CPU_FEATURE_AVX512F]  = (reg[1] & (1 << 16)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_AVX512CD] = (reg[1] & (1 << 28)) != 0;
    if (npy__cpu_have[NPY_CPU_FEATURE_AVX512F] && npy__cpu_have[NPY_CPU_FEATURE_AVX512CD]) {
        // Knights Landing
        npy__cpu_have[NPY_CPU_FEATURE_AVX512PF]        = (reg[1] & (1 << 26)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512ER]        = (reg[1] & (1 << 27)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_KNL]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512ER] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512PF];
        // Knights Mill
        npy__cpu_have[NPY_CPU_FEATURE_AVX512VPOPCNTDQ] = (reg[2] & (1 << 14)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX5124VNNIW]    = (reg[3] & (1 << 2))  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX5124FMAPS]    = (reg[3] & (1 << 3))  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_KNM]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512_KNL] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX5124FMAPS] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX5124VNNIW] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VPOPCNTDQ];

        // Skylake-X
        npy__cpu_have[NPY_CPU_FEATURE_AVX512DQ]        = (reg[1] & (1 << 17)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512BW]        = (reg[1] & (1 << 30)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512VL]        = (reg[1] & (1 << 31)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_SKX]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512BW] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512DQ] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VL];
        // Cascade Lake
        npy__cpu_have[NPY_CPU_FEATURE_AVX512VNNI]      = (reg[2] & (1 << 11)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_CLX]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512_SKX] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VNNI];

        // Cannon Lake
        npy__cpu_have[NPY_CPU_FEATURE_AVX512IFMA]      = (reg[1] & (1 << 21)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512VBMI]      = (reg[2] & (1 << 1))  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_CNL]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512_SKX] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512IFMA] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VBMI];
        // Ice Lake
        npy__cpu_have[NPY_CPU_FEATURE_AVX512VBMI2]     = (reg[2] & (1 << 6))  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512BITALG]    = (reg[2] & (1 << 12)) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_ICL]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512_CLX] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512_CNL] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VBMI2] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512BITALG] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512VPOPCNTDQ];
        // Sapphire Rapids
        npy__cpu_have[NPY_CPU_FEATURE_AVX512FP16]     = (reg[3] & (1 << 23))  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_AVX512_SPR]      = npy__cpu_have[NPY_CPU_FEATURE_AVX512_ICL] &&
                                                         npy__cpu_have[NPY_CPU_FEATURE_AVX512FP16];

    }
}

/***************** POWER ******************/

#elif defined(NPY_CPU_PPC64) || defined(NPY_CPU_PPC64LE)

#if defined(__linux__) || defined(__FreeBSD__)
    #ifdef __FreeBSD__
        #include <machine/cpu.h> // defines PPC_FEATURE_HAS_VSX
    #endif
    #include <sys/auxv.h>
    #ifndef AT_HWCAP2
        #define AT_HWCAP2 26
    #endif
    #ifndef PPC_FEATURE2_ARCH_2_07
        #define PPC_FEATURE2_ARCH_2_07 0x80000000
    #endif
    #ifndef PPC_FEATURE2_ARCH_3_00
        #define PPC_FEATURE2_ARCH_3_00 0x00800000
    #endif
    #ifndef PPC_FEATURE2_ARCH_3_1
        #define PPC_FEATURE2_ARCH_3_1  0x00040000
    #endif
#endif

static void
npy__cpu_init_features(void)
{
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
#if defined(__linux__) || defined(__FreeBSD__)
#ifdef __linux__
    unsigned int hwcap = getauxval(AT_HWCAP);
    if ((hwcap & PPC_FEATURE_HAS_VSX) == 0)
        return;

    hwcap = getauxval(AT_HWCAP2);
#else
    unsigned long hwcap;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    if ((hwcap & PPC_FEATURE_HAS_VSX) == 0)
        return;

    elf_aux_info(AT_HWCAP2, &hwcap, sizeof(hwcap));
#endif // __linux__
    if (hwcap & PPC_FEATURE2_ARCH_3_1)
    {
        npy__cpu_have[NPY_CPU_FEATURE_VSX]  =
        npy__cpu_have[NPY_CPU_FEATURE_VSX2] =
        npy__cpu_have[NPY_CPU_FEATURE_VSX3] =
        npy__cpu_have[NPY_CPU_FEATURE_VSX4] = 1;
        return;
    }
    npy__cpu_have[NPY_CPU_FEATURE_VSX]  = 1;
    npy__cpu_have[NPY_CPU_FEATURE_VSX2] = (hwcap & PPC_FEATURE2_ARCH_2_07) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_VSX3] = (hwcap & PPC_FEATURE2_ARCH_3_00) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_VSX4] = (hwcap & PPC_FEATURE2_ARCH_3_1) != 0;
// TODO: AIX, OpenBSD
#else
    npy__cpu_have[NPY_CPU_FEATURE_VSX]  = 1;
    #if defined(NPY_CPU_PPC64LE) || defined(NPY_HAVE_VSX2)
    npy__cpu_have[NPY_CPU_FEATURE_VSX2] = 1;
    #endif
    #ifdef NPY_HAVE_VSX3
    npy__cpu_have[NPY_CPU_FEATURE_VSX3] = 1;
    #endif
    #ifdef NPY_HAVE_VSX4
    npy__cpu_have[NPY_CPU_FEATURE_VSX4] = 1;
    #endif
#endif
}

/***************** ZARCH ******************/

#elif defined(__s390x__)

#include <sys/auxv.h>

/* kernel HWCAP names, available in musl, not available in glibc<2.33: https://sourceware.org/bugzilla/show_bug.cgi?id=25971 */
#ifndef HWCAP_S390_VXRS
    #define HWCAP_S390_VXRS 2048
#endif
#ifndef HWCAP_S390_VXRS_EXT
    #define HWCAP_S390_VXRS_EXT 8192
#endif
#ifndef HWCAP_S390_VXRS_EXT2
    #define HWCAP_S390_VXRS_EXT2 32768
#endif

static void
npy__cpu_init_features(void)
{
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

    unsigned int hwcap = getauxval(AT_HWCAP);
    if ((hwcap & HWCAP_S390_VXRS) == 0) {
        return;
    }

    if (hwcap & HWCAP_S390_VXRS_EXT2) {
       npy__cpu_have[NPY_CPU_FEATURE_VX]  =
       npy__cpu_have[NPY_CPU_FEATURE_VXE] =
       npy__cpu_have[NPY_CPU_FEATURE_VXE2] = 1;
       return;
    }

    npy__cpu_have[NPY_CPU_FEATURE_VXE] = (hwcap & HWCAP_S390_VXRS_EXT) != 0;

    npy__cpu_have[NPY_CPU_FEATURE_VX]  = 1;
}

/***************** LoongArch ******************/

#elif defined(__loongarch_lp64)

#include <sys/auxv.h>
#include <asm/hwcap.h>

static void
npy__cpu_init_features(void)
{
   memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
   unsigned int hwcap = getauxval(AT_HWCAP);

   if ((hwcap & HWCAP_LOONGARCH_LSX)) {
      npy__cpu_have[NPY_CPU_FEATURE_LSX]  = 1;
      return;
   }
}

/***************** ARM ******************/

#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)

static inline void
npy__cpu_init_features_arm8(void)
{
    npy__cpu_have[NPY_CPU_FEATURE_NEON]       =
    npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16]  =
    npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] =
    npy__cpu_have[NPY_CPU_FEATURE_ASIMD]      = 1;
}

#if defined(__linux__) || defined(__FreeBSD__)
/*
 * we aren't sure of what kind kernel or clib we deal with
 * so we play it safe
*/
#include <stdio.h>
#include "npy_cpuinfo_parser.h"

#if defined(__linux__)
__attribute__((weak)) unsigned long getauxval(unsigned long); // linker should handle it
#endif
#ifdef __FreeBSD__
__attribute__((weak)) int elf_aux_info(int, void *, int); // linker should handle it

static unsigned long getauxval(unsigned long k)
{
    unsigned long val = 0ul;
    if (elf_aux_info == 0 || elf_aux_info((int)k, (void *)&val, (int)sizeof(val)) != 0) {
    	return 0ul;
    }
    return val;
}
#endif
static int
npy__cpu_init_features_linux(void)
{
    unsigned long hwcap = 0, hwcap2 = 0;
    #ifdef __linux__
    if (getauxval != 0) {
        hwcap = getauxval(NPY__HWCAP);
    #ifdef __arm__
        hwcap2 = getauxval(NPY__HWCAP2);
    #endif
    } else {
        unsigned long auxv[2];
        int fd = open("/proc/self/auxv", O_RDONLY);
        if (fd >= 0) {
            while (read(fd, &auxv, sizeof(auxv)) == sizeof(auxv)) {
                if (auxv[0] == NPY__HWCAP) {
                    hwcap = auxv[1];
                }
            #ifdef __arm__
                else if (auxv[0] == NPY__HWCAP2) {
                    hwcap2 = auxv[1];
                }
            #endif
                // detect the end
                else if (auxv[0] == 0 && auxv[1] == 0) {
                    break;
                }
            }
            close(fd);
        }
    }
    #else
    hwcap = getauxval(NPY__HWCAP);
    #ifdef __arm__
    hwcap2 = getauxval(NPY__HWCAP2);
    #endif
    #endif
    if (hwcap == 0 && hwcap2 == 0) {
    #ifdef __linux__
        /*
         * try parsing with /proc/cpuinfo, if sandboxed
         * failback to compiler definitions
        */
        if(!get_feature_from_proc_cpuinfo(&hwcap, &hwcap2)) {
            return 0;
        }
    #else
    	return 0;
    #endif
    }
#ifdef __arm__
    npy__cpu_have[NPY_CPU_FEATURE_NEON]       = (hwcap & NPY__HWCAP_NEON)   != 0;
    if (npy__cpu_have[NPY_CPU_FEATURE_NEON]) {
        npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16]  = (hwcap & NPY__HWCAP_HALF) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] = (hwcap & NPY__HWCAP_VFPv4) != 0;
    }
    // Detect Arm8 (aarch32 state)
    if ((hwcap2 & NPY__HWCAP2_AES)  || (hwcap2 & NPY__HWCAP2_SHA1)  ||
        (hwcap2 & NPY__HWCAP2_SHA2) || (hwcap2 & NPY__HWCAP2_PMULL) ||
        (hwcap2 & NPY__HWCAP2_CRC32))
    {
        npy__cpu_have[NPY_CPU_FEATURE_ASIMD] = npy__cpu_have[NPY_CPU_FEATURE_NEON];
    }
#else
    if (!(hwcap & (NPY__HWCAP_FP | NPY__HWCAP_ASIMD))) {
        // Is this could happen? maybe disabled by kernel
        // BTW this will break the baseline of AARCH64
        return 1;
    }
    npy__cpu_init_features_arm8();
#endif
    npy__cpu_have[NPY_CPU_FEATURE_FPHP]       = (hwcap & NPY__HWCAP_FPHP)     != 0;
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDHP]    = (hwcap & NPY__HWCAP_ASIMDHP)  != 0;
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDDP]    = (hwcap & NPY__HWCAP_ASIMDDP)  != 0;
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDFHM]   = (hwcap & NPY__HWCAP_ASIMDFHM) != 0;
#ifndef __arm__
    npy__cpu_have[NPY_CPU_FEATURE_SVE]        = (hwcap & NPY__HWCAP_SVE)      != 0;
#endif
    return 1;
}
#endif

static void
npy__cpu_init_features(void)
{
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
#ifdef __linux__
    if (npy__cpu_init_features_linux())
        return;
#endif
    // We have nothing else todo
#if defined(NPY_HAVE_ASIMD) || defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 8) || defined(_M_ARM64)
    #if defined(NPY_HAVE_FPHP) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    npy__cpu_have[NPY_CPU_FEATURE_FPHP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDHP) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDHP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDDP) || defined(__ARM_FEATURE_DOTPROD)
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDDP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDFHM) || defined(__ARM_FEATURE_FP16FML)
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDFHM] = 1;
    #endif
    #if defined(NPY_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
    npy__cpu_have[NPY_CPU_FEATURE_SVE] = 1;
    #endif
    npy__cpu_init_features_arm8();
#else
    #if defined(NPY_HAVE_NEON) || defined(__ARM_NEON__)
        npy__cpu_have[NPY_CPU_FEATURE_NEON] = 1;
    #endif
    #if defined(NPY_HAVE_NEON_FP16) || defined(__ARM_FP16_FORMAT_IEEE) || (defined(__ARM_FP) && (__ARM_FP & 2))
        npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16] = npy__cpu_have[NPY_CPU_FEATURE_NEON];
    #endif
    #if defined(NPY_HAVE_NEON_VFPV4) || defined(__ARM_FEATURE_FMA)
        npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] = npy__cpu_have[NPY_CPU_FEATURE_NEON];
    #endif
#endif
}

/************** RISC-V 64 ***************/

#elif defined(__riscv) && __riscv_xlen == 64

#include <sys/auxv.h>

#ifndef HWCAP_RVV
    // https://github.com/torvalds/linux/blob/v6.8/arch/riscv/include/uapi/asm/hwcap.h#L24
    #define COMPAT_HWCAP_ISA_V	(1 << ('V' - 'A'))
#endif

static void
npy__cpu_init_features(void)
{
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

    unsigned int hwcap = getauxval(AT_HWCAP);
    if (hwcap & COMPAT_HWCAP_ISA_V) {
        npy__cpu_have[NPY_CPU_FEATURE_RVV]  = 1;
    }
}

/*********** Unsupported ARCH ***********/
#else
static void
npy__cpu_init_features(void)
{
    /*
     * just in case if the compiler doesn't respect ANSI
     * but for knowing platforms it still necessary, because @npy__cpu_init_features
     * may called multiple of times and we need to clear the disabled features by
     * ENV Var or maybe in the future we can support other methods like
     * global variables, go back to @npy__cpu_try_disable_env for more understanding
     */
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
}
#endif

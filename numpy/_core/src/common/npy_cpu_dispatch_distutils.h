#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_H_
    #error "Not standalone header please use 'npy_cpu_dispatch.h'"
#endif
/**
 * This header should be removed after support for distutils is removed.
 * It provides helper macros required for CPU runtime dispatching,
 * which are already defined within `meson_cpu/main_config.h.in`.
 *
 * The following macros are explained within `meson_cpu/main_config.h.in`,
 * although there are some differences in their usage:
 *
 * - Dispatched targets must be defined at the top of each dispatch-able
 *   source file within an inline or multi-line comment block.
 *   For example: //@targets baseline SSE2 AVX2 AVX512_SKX
 *
 * - The generated configuration derived from each dispatch-able source
 *   file must be guarded with `#ifndef NPY_DISABLE_OPTIMIZATION`.
 *   For example:
 *   #ifndef NPY_DISABLE_OPTIMIZATION
 *      #include "arithmetic.dispatch.h"
 *   #endif
 */
#include "npy_cpu_features.h" // NPY_CPU_HAVE
#include "numpy/utils.h" // NPY_EXPAND, NPY_CAT

#ifdef NPY__CPU_TARGET_CURRENT
    // 'NPY__CPU_TARGET_CURRENT': only defined by the dispatch-able sources
    #define NPY_CPU_DISPATCH_CURFX(NAME) NPY_CAT(NPY_CAT(NAME, _), NPY__CPU_TARGET_CURRENT)
#else
    #define NPY_CPU_DISPATCH_CURFX(NAME) NPY_EXPAND(NAME)
#endif
/**
 * Defining the default behavior for the configurable macros of dispatch-able sources,
 * 'NPY__CPU_DISPATCH_CALL(...)' and 'NPY__CPU_DISPATCH_BASELINE_CALL(...)'
 *
 * These macros are defined inside the generated config files that been derived from
 * the configuration statements of the dispatch-able sources.
 *
 * The generated config file takes the same name of the dispatch-able source with replacing
 * the extension to '.h' instead of '.c', and it should be treated as a header template.
 */
#ifndef NPY_DISABLE_OPTIMIZATION
    #define NPY__CPU_DISPATCH_BASELINE_CALL(CB, ...) \
        &&"Expected config header of the dispatch-able source";
    #define NPY__CPU_DISPATCH_CALL(CHK, CB, ...) \
        &&"Expected config header of the dispatch-able source";
#else
    /**
     * We assume by default that all configuration statements contains 'baseline' option however,
     * if the dispatch-able source doesn't require it, then the dispatch-able source and following macros
     * need to be guard it with '#ifndef NPY_DISABLE_OPTIMIZATION'
     */
    #define NPY__CPU_DISPATCH_BASELINE_CALL(CB, ...) \
        NPY_EXPAND(CB(__VA_ARGS__))
    #define NPY__CPU_DISPATCH_CALL(CHK, CB, ...)
#endif // !NPY_DISABLE_OPTIMIZATION

#define NPY_CPU_DISPATCH_DECLARE(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_DISPATCH_DECLARE_CHK_, NPY_CPU_DISPATCH_DECLARE_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_DECLARE_BASE_CB_, __VA_ARGS__)
// Preprocessor callbacks
#define NPY_CPU_DISPATCH_DECLARE_CB_(DUMMY, TARGET_NAME, LEFT, ...) \
    NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__;
#define NPY_CPU_DISPATCH_DECLARE_BASE_CB_(LEFT, ...) \
    LEFT __VA_ARGS__;
// Dummy CPU runtime checking
#define NPY_CPU_DISPATCH_DECLARE_CHK_(FEATURE)

#define NPY_CPU_DISPATCH_DECLARE_XB(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_DISPATCH_DECLARE_CHK_, NPY_CPU_DISPATCH_DECLARE_CB_, __VA_ARGS__)
#define NPY_CPU_DISPATCH_CALL(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_CALL_BASE_CB_, __VA_ARGS__)
// Preprocessor callbacks
#define NPY_CPU_DISPATCH_CALL_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    (TESTED_FEATURES) ? (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) :
#define NPY_CPU_DISPATCH_CALL_BASE_CB_(LEFT, ...) \
    (LEFT __VA_ARGS__)

#define NPY_CPU_DISPATCH_CALL_XB(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_XB_CB_, __VA_ARGS__) \
    ((void) 0 /* discarded expression value */)
#define NPY_CPU_DISPATCH_CALL_XB_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    (TESTED_FEATURES) ? (void) (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) :

#define NPY_CPU_DISPATCH_CALL_ALL(...) \
    (NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_ALL_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_CALL_ALL_BASE_CB_, __VA_ARGS__))
// Preprocessor callbacks
#define NPY_CPU_DISPATCH_CALL_ALL_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    ((TESTED_FEATURES) ? (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) : (void) 0),
#define NPY_CPU_DISPATCH_CALL_ALL_BASE_CB_(LEFT, ...) \
    ( LEFT __VA_ARGS__ )

#define NPY_CPU_DISPATCH_INFO() \
    { \
        NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_INFO_HIGH_CB_, DUMMY) \
        NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_INFO_BASE_HIGH_CB_, DUMMY) \
        "", \
        NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_INFO_CB_, DUMMY) \
        NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_INFO_BASE_CB_, DUMMY) \
        ""\
    }
#define NPY_CPU_DISPATCH_INFO_HIGH_CB_(TESTED_FEATURES, TARGET_NAME, ...) \
    (TESTED_FEATURES) ? NPY_TOSTRING(TARGET_NAME) :
#define NPY_CPU_DISPATCH_INFO_BASE_HIGH_CB_(...) \
    (1) ? "baseline(" NPY_WITH_CPU_BASELINE ")" :
// Preprocessor callbacks
#define NPY_CPU_DISPATCH_INFO_CB_(TESTED_FEATURES, TARGET_NAME, ...) \
    NPY_TOSTRING(TARGET_NAME) " "
#define NPY_CPU_DISPATCH_INFO_BASE_CB_(...) \
    "baseline(" NPY_WITH_CPU_BASELINE ")"

#endif  // NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_

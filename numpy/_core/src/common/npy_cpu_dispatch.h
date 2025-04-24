#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_H_
/**
 * This file is part of the NumPy CPU dispatcher.
 *
 * Please have a look at doc/reference/simd-optimizations.html
 * To get a better understanding of the mechanism behind it.
 */
#include "npy_cpu_features.h" // NPY_CPU_HAVE
/**
 *  This header genereated by the build system and contains:
 *
 *   - Headers for platform-specific instruction sets.
 *   - Helper macros that encapsulate enabled features through user-defined build options
 *     '--cpu-baseline' and '--cpu-dispatch'. These options remain crucial for implementing
 *     attributes like `__cpu_baseline__` and `__cpu_dispatch__` in the NumPy module.
 *   - Additional helper macros necessary for runtime dispatching.
 *
 * Note: features #definitions are conveyed via compiler arguments.
 */
#include "npy_cpu_dispatch_config.h"

/**
 * Initialize the CPU dispatch tracer.
 *
 * This function simply adds an empty dictionary with the attribute
 * '__cpu_targets_info__' to the provided module.
 *
 * It should be called only once during the loading of the NumPy module.
 * Note: This function is not thread-safe.
 *
 * @param mod The module to which the '__cpu_targets_info__' dictionary will be added.
 * @return 0 on success.
 */
NPY_VISIBILITY_HIDDEN int
npy_cpu_dispatch_tracer_init(PyObject *mod);
/**
 * Insert data into the initialized '__cpu_targets_info__' dictionary.
 *
 * This function adds the function name as a key and another dictionary as a value.
 * The inner dictionary holds the 'signature' as a key and splits 'dispatch_info' into another dictionary.
 * The innermost dictionary contains the current enabled target as 'current' and available targets as 'available'.
 *
 * Note: This function should not be used directly; it should be used through the macro NPY_CPU_DISPATCH_TRACE(),
 * which is responsible for filling in the enabled CPU targets.
 *
 * Example:
 *
 * const char *dispatch_info[] = {"AVX2", "AVX512_SKX AVX2 baseline"};
 * npy_cpu_dispatch_trace("add", "bbb", dispatch_info);
 *
 * const char *dispatch_info[] = {"AVX2", "AVX2 SSE41 baseline"};
 * npy_cpu_dispatch_trace("add", "BBB", dispatch_info);
 *
 * This will insert the following structure into the '__cpu_targets_info__' dictionary:
 *
 * numpy._core._multiarray_umath.__cpu_targets_info__
 * {
 *    "add": {
 *      "bbb": {
 *        "current": "AVX2",
 *        "available": "AVX512_SKX AVX2 baseline"
 *      },
 *      "BBB": {
 *        "current": "AVX2",
 *        "available": "AVX2 SSE41 baseline"
 *      },
 *    },
 * }
 *
 * @param func_name The name of the function.
 * @param signature The signature of the function.
 * @param dispatch_info The information about CPU dispatching.
 */
NPY_VISIBILITY_HIDDEN void
npy_cpu_dispatch_trace(const char *func_name, const char *signature,
                       const char **dispatch_info);
/**
 * Extract the enabled CPU targets from the generated configuration file.
 *
 * This macro is used to extract the enabled CPU targets from the generated configuration file,
 * which is derived from 'meson.multi_targets()' or from 'disutils.CCompilerOpt' in the case of using distutils.
 * It then calls 'npy_cpu_dispatch_trace()' to insert a new item into the '__cpu_targets_info__' dictionary,
 * based on the provided FUNC_NAME and SIGNATURE.
 *
 * For more clarification, please refer to the macro 'NPY_CPU_DISPATCH_INFO()' defined in 'meson_cpu/main_config.h.in'
 * and check 'np.lib.utils.opt_func_info()' for the final usage of this trace.
 *
 * Example:
 * #include "arithmetic.dispatch.h"
 * NPY_CPU_DISPATCH_CALL(BYTE_add_ptr = BYTE_add);
 * NPY_CPU_DISPATCH_TRACE("add", "bbb");
 */
#define NPY_CPU_DISPATCH_TRACE(FNAME, SIGNATURE)     \
{                                                    \
    const char *dinfo[] = NPY_CPU_DISPATCH_INFO();   \
    npy_cpu_dispatch_trace(FNAME, SIGNATURE, dinfo); \
} while(0)

#endif  // NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_H_

#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A struct storing global state for the _multiarray_umath
 * module. The state is initialized when the module is imported
 * so no locking is necessary to access it.
 *
 * These globals will need to move to per-module state to
 * support reloading or subinterpreters.
 */
typedef struct npy_global_state_struct {
    /*
     * Used to test the internal-only scaled float test dtype
     */
    npy_bool get_sfloat_dtype_initialized;

    /*
     * controls the global madvise hugepage setting
     */
    int madvise_hugepage;

    /*
     * used to detect module reloading in the reload guard
     */
    int reload_guard_initialized;

    /*
     * Holds the user-defined setting for whether or not to warn
     * if there is no memory policy set
     */
    int warn_if_no_mem_policy;
} npy_global_state_struct;


NPY_VISIBILITY_HIDDEN extern npy_global_state_struct npy_global_state;

NPY_NO_EXPORT int
get_legacy_print_mode(void);
#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

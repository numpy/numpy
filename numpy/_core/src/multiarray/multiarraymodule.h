#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

/*
 * Struct definition for per-module global state, embedded as the
 * global_state field of multiarray_umath_state (see module_state.h).
 * Python allocates one instance per interpreter; access it via
 * get_module_state() or npy_get_module_state().
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

NPY_NO_EXPORT int
get_legacy_print_mode(void);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

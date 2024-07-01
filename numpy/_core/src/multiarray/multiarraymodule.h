#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

/*
 * A struct storing thread-unsafe global state for the _multiarray_umath
 * module. We should refactor so the global state is thread-safe,
 * e.g. by adding locking.
 */
typedef struct npy_thread_unsafe_state_struct {
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
      * global variable to determine if legacy printing is enabled,
      * accessible from C. For simplicity the mode is encoded as an
      * integer where INT_MAX means no legacy mode, and '113'/'121'
      * means 1.13/1.21 legacy mode; and 0 maps to INT_MAX. We can
      * upgrade this if we have more complex requirements in the future.
      */
    int legacy_print_mode;

    /*
     * Holds the user-defined setting for whether or not to warn
     * if there is no memory policy set
     */
    int warn_if_no_mem_policy;

} npy_thread_unsafe_state_struct;


NPY_VISIBILITY_HIDDEN extern npy_thread_unsafe_state_struct npy_thread_unsafe_state;


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

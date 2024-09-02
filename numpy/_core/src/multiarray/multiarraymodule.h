#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

/*
 * A struct storing thread-unsafe global state for the _multiarray_umath
 * module. We should refactor so the global state is thread-safe,
 * e.g. by adding locking.
 */
typedef struct npy_thread_unsafe_state_struct {
    /*
     * Cached references to objects obtained via an import. All of these are
     * can be initialized at any time by npy_cache_import.
     *
     * Currently these are not initialized in a thread-safe manner but the
     * failure mode is a reference leak for references to imported immortal
     * modules so it will never lead to a crash unless users are doing something
     * janky that we don't support like reloading.
     *
     * TODO: maybe make each entry a struct that looks like:
     *
     *      struct {
     *          atomic_int initialized;
     *          PyObject *value;
     *      }
     *
     * so the initialization is thread-safe and the only possible lock
     * contention happens before the cache is initialized, not on every single
     * read.
     */
    PyObject *_add_dtype_helper;
    PyObject *_all;
    PyObject *_amax;
    PyObject *_amin;
    PyObject *_any;
    PyObject *array_function_errmsg_formatter;
    PyObject *array_ufunc_errmsg_formatter;
    PyObject *_clip;
    PyObject *_commastring;
    PyObject *_convert_to_stringdtype_kwargs;
    PyObject *_default_array_repr;
    PyObject *_default_array_str;
    PyObject *_dump;
    PyObject *_dumps;
    PyObject *_getfield_is_safe;
    PyObject *internal_gcd_func;
    PyObject *_mean;
    PyObject *NO_NEP50_WARNING;
    PyObject *npy_ctypes_check;
    PyObject *numpy_matrix;
    PyObject *_prod;
    PyObject *_promote_fields;
    PyObject *_std;
    PyObject *_sum;
    PyObject *_ufunc_doc_signature_formatter;
    PyObject *_var;
    PyObject *_view_is_safe;
    PyObject *_void_scalar_to_string;

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

} npy_thread_unsafe_state_struct;


NPY_VISIBILITY_HIDDEN extern npy_thread_unsafe_state_struct npy_thread_unsafe_state;

NPY_NO_EXPORT int
get_legacy_print_mode(void);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

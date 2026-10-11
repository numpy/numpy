#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Embedded as the global_state field of multiarray_umath_state. */
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

#define NPY_DOT_REPLACEMENT "numpy.tensordot(a, b, axes=[-1, -2])"
#define NPY_INNER_REPLACEMENT "numpy.tensordot(a, b, axes=[-1, -1])"

/* Internal matrix product; warns as `funcname` when a.ndim >= 2 and b.ndim > 2. */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct_int(PyObject *op1, PyObject *op2, PyArrayObject *out,
                    const char *funcname, const char *replacement);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

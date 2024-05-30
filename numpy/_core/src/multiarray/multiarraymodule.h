#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

typedef struct npy_ma_str_struct {
    PyObject *current_allocator;
    PyObject *array;
    PyObject *array_function;
    PyObject *array_struct;
    PyObject *array_priority;
    PyObject *array_interface;
    PyObject *array_wrap;
    PyObject *array_finalize;
    PyObject *implementation;
    PyObject *axis1;
    PyObject *axis2;
    PyObject *like;
    PyObject *numpy;
    PyObject *where;
    PyObject *convert;
    PyObject *preserve;
    PyObject *convert_if_no_array;
    PyObject *cpu;
    PyObject *dtype;
    PyObject *array_err_msg_substr;
    PyObject *out;
    PyObject *errmode_strings[6];
    PyObject *__dlpack__;
} npy_ma_str_struct;

NPY_VISIBILITY_HIDDEN extern npy_ma_str_struct *npy_ma_str;

typedef struct npy_ma_global_data_struct {
    /*
     * Used in ufunc_type_resolution.c to avoid reconstructing a tuple
     * storing the default true division return types
     * This is immutable and set at module initialization so can be used
     * without acquiring the global data mutex
     */
    PyObject *default_truediv_type_tup;

    /*
     * Used to set up the default extobj context variable
     *
     * This is immutable and set at module initialization so can be used
     * without acquiring the global data mutex
     */
    PyObject *default_extobj_capsule;

    /*
     * The global ContextVar to store the extobject. It is exposed to Python
     * as `_extobj_contextvar`.
     */
    PyObject *npy_extobj_contextvar;

    /*
     * A reference to ndarray's implementations for __array_*__ special methods
     */
    PyObject *ndarray_array_ufunc;
    PyObject *ndarray_array_finalize;
    PyObject *ndarray_array_function;

    /*
     * References to the '1' and '0' PyLong objects
     */
    PyObject *one_obj;
    PyObject *zero_obj;

    /*
     * References to items obtained via an import at module initialization
     *
     * These are immutable
     */
    PyObject *AxisError;
    PyObject *ComplexWarning;
    PyObject *DTypePromotionError;
    PyObject *TooHardError;
    PyObject *VisibleDeprecationWarning;
    PyObject *_CopyMode;
    PyObject *_NoValue;
    PyObject *_ArrayMemoryError;
    PyObject *_UFuncBinaryResolutionError;
    PyObject *_UFuncInputCastingError;
    PyObject *_UFuncNoLoopError;
    PyObject *_UFuncOutputCastingError;
    PyObject *math_floor_func;
    PyObject *math_ceil_func;
    PyObject *math_trunc_func;
    PyObject *math_gcd_func;
    PyObject *os_PathLike;
    PyObject *os_fspath;

    /*
     * stores sys.flags.optimize as a long, which is used in the add_docstring
     * implementation
     */
    long optimize;

    /*
     * LUT used by unpack_bits
     */
    union {
        npy_uint8  bytes[8];
        npy_uint64 uint64;
    } unpack_lookup_big[256];

    /*
     * Used in the __array__ internals to avoid building a tuple inline
     */
    PyObject *kwnames_is_copy;

    /*
     * Used in __imatmul__ to avoid building tuples inline
     */
    PyObject *axes_1d_obj_kwargs;
    PyObject *axes_2d_obj_kwargs;

    /*
     * Used for CPU feature detection and dispatch
     *
     * Filled in during module initialization and thereafter immutable
     */
    PyObject *cpu_dispatch_registry;

    /*
     * The following entries store cached references to object obtained
     * via an import. All of these are initialized at runtime by
     * npy_cache_import.
     *
     * Currently these are not initialized in a thread-safe manner but the
     * failure mode is a reference leak for references to imported modules so
     * it will never lead to a crash unless there is something janky that we
     * don't support going on like reloading.
     *
     * TODO: maybe make each entry a struct that looks like:
     *
     *      struct {
     *          atomic_int initialized;
     *          PyObject *value;
     *      }
     *
     * so is thread-safe initialization and only the possibility of contention
     * before the cache is initialized, not on every single read.
     */
    PyObject *_add_dtype_helper;
    PyObject *_all;
    PyObject *_amax;
    PyObject *_amin;
    PyObject *_any;
    PyObject *_clip;
    PyObject *_commastring;
    PyObject *_convert_to_stringdtype_kwargs;
    PyObject *_default_array_repr;
    PyObject *_default_array_str;
    PyObject *_dump;
    PyObject *_dumps;
    PyObject *_getfield_is_safe;
    PyObject *_mean;
    PyObject *_prod;
    PyObject *_promote_fields;
    PyObject *_std;
    PyObject *_sum;
    PyObject *_ufunc_doc_signature_formatter;
    PyObject *_var;
    PyObject *_view_is_safe;
    PyObject *_void_scalar_to_string;
    PyObject *array_function_errmsg_formatter;
    PyObject *array_ufunc_errmsg_formatter;
    PyObject *internal_gcd_func;
    PyObject *npy_ctypes_check;
    PyObject *numpy_matrix;
    PyObject *NO_NEP50_WARNING;
} npy_ma_global_data_struct;

NPY_VISIBILITY_HIDDEN extern npy_ma_global_data_struct *npy_ma_global_data;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

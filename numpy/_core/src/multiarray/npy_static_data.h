#ifndef NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
initialize_static_globals(void);

NPY_NO_EXPORT int
intern_strings(void);

NPY_NO_EXPORT int
verify_static_structs_initialized(void);

typedef struct npy_interned_str_struct {
    PyObject *current_allocator;
    PyObject *array;
    PyObject *array_function;
    PyObject *array_struct;
    PyObject *array_priority;
    PyObject *array_interface;
    PyObject *array_wrap;
    PyObject *array_finalize;
    PyObject *array_ufunc;
    PyObject *implementation;
    PyObject *axis1;
    PyObject *axis2;
    PyObject *item;
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
    PyObject *pyvals_name;
    PyObject *legacy;
    PyObject *__doc__;
    PyObject *copy;
    PyObject *dl_device;
    PyObject *max_version;
} npy_interned_str_struct;

/*
 * A struct that stores static global data used throughout
 * _multiarray_umath, mostly to cache results that would be
 * prohibitively expensive to compute at runtime in a tight loop.
 *
 * All items in this struct should be initialized during module
 * initialization and thereafter should be immutable. Mutating items in
 * this struct after module initialization is likely not thread-safe.
 */

typedef struct npy_static_pydata_struct {
    /*
     * Used in ufunc_type_resolution.c to avoid reconstructing a tuple
     * storing the default true division return types.
     */
    PyObject *default_truediv_type_tup;

    /*
     * Used to set up the default extobj context variable
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
     * Reference to an np.array(0, dtype=np.long) instance
     */
    PyObject *zero_pyint_like_arr;

    /*
     * References to items obtained via an import at module initialization
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
    PyObject *format_options;

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
     */
    PyObject *cpu_dispatch_registry;

    /*
     * references to ArrayMethod implementations that are cached
     * to avoid repeatedly creating them
     */
    PyObject *VoidToGenericMethod;
    PyObject *GenericToVoidMethod;
    PyObject *ObjectToGenericMethod;
    PyObject *GenericToObjectMethod;

    /*
     * Used in from_dlpack
     */
    PyObject *dl_call_kwnames;
    PyObject *dl_cpu_device_tuple;
    PyObject *dl_max_version;
} npy_static_pydata_struct;


typedef struct npy_static_cdata_struct {
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
     * A look-up table to recover integer type numbers from type characters.
     *
     * See the _MAX_LETTER and LETTER_TO_NUM macros in arraytypes.c.src.
     *
     * The smallest type number is ?, the largest is bounded by 'z'.
     *
     * This is initialized alongside the built-in dtypes
     */
    npy_int16 _letter_to_num['z' + 1 - '?'];
} npy_static_cdata_struct;

NPY_VISIBILITY_HIDDEN extern npy_interned_str_struct npy_interned_str;
NPY_VISIBILITY_HIDDEN extern npy_static_pydata_struct npy_static_pydata;
NPY_VISIBILITY_HIDDEN extern npy_static_cdata_struct npy_static_cdata;

#ifdef __cplusplus
}
#endif

#endif  // NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_

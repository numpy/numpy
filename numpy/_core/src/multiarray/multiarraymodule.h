#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

typedef struct npy_interned_str_struct {
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
     */
    npy_int16 _letter_to_num['z' + 1 - '?'];
} npy_static_cdata_struct;

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
     * so the initialization is thread-safe and the only possibile lock
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
} npy_thread_unsafe_state_struct;


NPY_VISIBILITY_HIDDEN extern npy_interned_str_struct npy_interned_str;
NPY_VISIBILITY_HIDDEN extern npy_static_pydata_struct npy_static_pydata;
NPY_VISIBILITY_HIDDEN extern npy_static_cdata_struct npy_static_cdata;
NPY_VISIBILITY_HIDDEN extern npy_thread_unsafe_state_struct npy_thread_unsafe_state;


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

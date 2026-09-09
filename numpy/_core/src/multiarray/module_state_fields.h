#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_

/*
 * Field lists for the PyObject members of multiarray_umath_state, used to
 * expand the module traverse and clear functions.
 *
 * interned_str.errmode_strings is an array, so it is handled separately at
 * each use site.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define NPY_INTERNED_STR_FIELDS(F) \
    F(current_allocator)    \
    F(array)                \
    F(array_function)       \
    F(array_struct)         \
    F(array_priority)       \
    F(array_interface)      \
    F(array_wrap)           \
    F(array_finalize)       \
    F(array_ufunc)          \
    F(numpy_dtype)          \
    F(implementation)       \
    F(axis1)                \
    F(axis2)                \
    F(item)                 \
    F(like)                 \
    F(numpy)                \
    F(where)                \
    F(convert)              \
    F(preserve)             \
    F(convert_if_no_array)  \
    F(cpu)                  \
    F(dtype)                \
    F(array_err_msg_substr) \
    F(out)                  \
    F(__dlpack__)           \
    F(pyvals_name)          \
    F(legacy)               \
    F(__doc__)              \
    F(__signature__)        \
    F(copy)                 \
    F(dl_device)            \
    F(max_version)          \
    F(array_dealloc)        \
    F(real)                 \
    F(imag)                 \
    F(sort)                 \
    F(argsort)              \
    F(as_arrays)            \
    F(wrap)                 \
    F(subok)                \
    F(to_scalar)            \
    F(partition)            \
    F(argpartition)         \
    F(_set_dtype)           \
    F(conjugate)            \
    F(astimezone)           \
    F(value)                \
    F(year)                 \
    F(month)                \
    F(day)                  \
    F(hour)                 \
    F(minute)               \
    F(second)               \
    F(microsecond)          \
    F(tzinfo)               \
    F(utcoffset)            \
    F(total_seconds)        \
    F(reduce)               \
    F(accumulate)

#define NPY_STATIC_PYDATA_FIELDS(F)                                             \
    /*                                                                          \
     * Used in ufunc_type_resolution.c to avoid reconstructing a tuple          \
     * storing the default true division return types.                          \
     */                                                                         \
    F(default_truediv_type_tup)                                                 \
                                                                                \
    /*                                                                          \
     * Used to set up the default extobj context variable                       \
     */                                                                         \
    F(default_extobj_capsule)                                                   \
                                                                                \
    /*                                                                          \
     * The global ContextVar to store the extobject. It is exposed to Python    \
     * as `_extobj_contextvar`.                                                 \
     */                                                                         \
    F(npy_extobj_contextvar)                                                    \
                                                                                \
    /*                                                                          \
     * A reference to ndarray's implementations for __array_*__ special methods \
     */                                                                         \
    F(ndarray_array_ufunc)                                                      \
    F(ndarray_array_finalize)                                                   \
    F(ndarray_array_function)                                                   \
                                                                                \
    /*                                                                          \
     * References to ndarray._set_dtype and ndarray.dtype descriptor,           \
     * used in PyArray_View to detect subclass overrides.                       \
     */                                                                         \
    F(ndarray_set_dtype)                                                        \
    F(ndarray_dtype_descr)                                                      \
                                                                                \
    /*                                                                          \
     * References to the '1' and '0' PyLong objects                             \
     */                                                                         \
    F(one_obj)                                                                  \
    F(zero_obj)                                                                 \
                                                                                \
    /*                                                                          \
     * Reference to an np.array(0, dtype=np.long) instance                      \
     */                                                                         \
    F(zero_pyint_like_arr)                                                      \
                                                                                \
    /*                                                                          \
     * References to items obtained via an import at module initialization      \
     */                                                                         \
    F(AxisError)                                                                \
    F(ComplexWarning)                                                           \
    F(DTypePromotionError)                                                      \
    F(TooHardError)                                                             \
    F(VisibleDeprecationWarning)                                                \
    F(_CopyMode)                                                                \
    F(_NoValue)                                                                 \
    F(_ArrayMemoryError)                                                        \
    F(_UFuncBinaryResolutionError)                                              \
    F(_UFuncInputCastingError)                                                  \
    F(_UFuncNoLoopError)                                                        \
    F(_UFuncOutputCastingError)                                                 \
    F(math_floor_func)                                                          \
    F(math_ceil_func)                                                           \
    F(math_trunc_func)                                                          \
    F(math_gcd_func)                                                            \
    F(os_PathLike)                                                              \
    F(os_fspath)                                                                \
    F(format_options)                                                           \
                                                                                \
    /*                                                                          \
     * Context variable set to True while the legacy ufunc type resolvers       \
     * run for promotion, to suppress their deprecation warnings (the           \
     * resolution step warns on every call).                                    \
     */                                                                         \
    F(legacy_resolver_promoting)                                                \
                                                                                \
    /*                                                                          \
     * Used in the __array__ internals to avoid building a tuple inline         \
     */                                                                         \
    F(kwnames_is_copy)                                                          \
                                                                                \
    /*                                                                          \
     * Used by _wrapit to call the array converter's as_arrays/wrap             \
     * methods without building kwnames tuples inline                           \
     */                                                                         \
    F(wrapit_kwnames_subok)                                                     \
    F(wrapit_kwnames_to_scalar)                                                 \
                                                                                \
    F(kwnames_dtype)                                                            \
    F(kwnames_out)                                                              \
    F(kwnames_dtype_out)                                                        \
                                                                                \
    /*                                                                          \
     * Used in __imatmul__ to avoid building tuples inline                      \
     */                                                                         \
    F(axes_1d_obj_kwargs)                                                       \
    F(axes_2d_obj_kwargs)                                                       \
                                                                                \
    /*                                                                          \
     * Used for CPU feature detection and dispatch                              \
     */                                                                         \
    F(cpu_dispatch_registry)                                                    \
                                                                                \
    /*                                                                          \
     * references to ArrayMethod implementations that are cached                \
     * to avoid repeatedly creating them                                        \
     */                                                                         \
    F(VoidToGenericMethod)                                                      \
    F(GenericToVoidMethod)                                                      \
    F(ObjectToGenericMethod)                                                    \
    F(GenericToObjectMethod)                                                    \
                                                                                \
    /*                                                                          \
     * Used in from_dlpack                                                      \
     */                                                                         \
    F(dl_call_kwnames)                                                          \
    F(dl_cpu_device_tuple)                                                      \
    F(dl_max_version)                                                           \
    /* dicts for implementing `register_dlpack_dtype` */                        \
    F(dlpack_dtype_registry)                                                    \
    F(dlpack_export_registry)                                                   

#define NPY_RUNTIME_IMPORTS_FIELDS(F) \
    F(_add_dtype_helper)                \
    F(_all)                             \
    F(_amax)                            \
    F(_amin)                            \
    F(_any)                             \
    F(array_function_errmsg_formatter)  \
    F(array_ufunc_errmsg_formatter)     \
    F(_clip)                            \
    F(_commastring)                     \
    F(_convert_to_stringdtype_kwargs)   \
    F(_default_array_repr)              \
    F(_default_array_str)               \
    F(_dump)                            \
    F(_dumps)                           \
    F(_getfield_is_safe)                \
    F(internal_gcd_func)                \
    F(_mean)                            \
    F(NO_NEP50_WARNING)                 \
    F(npy_ctypes_check)                 \
    F(numpy_matrix)                     \
    F(_prod)                            \
    F(_promote_fields)                  \
    F(_std)                             \
    F(_sum)                             \
    F(_ufunc_doc_signature_formatter)   \
    F(_ufunc_inspect_signature_builder) \
    F(_usefields)                       \
    F(_var)                             \
    F(_view_is_safe)                    \
    F(_void_scalar_to_string)

#define NPY_N_OPS_FIELDS(F) \
    F(add)           \
    F(subtract)      \
    F(multiply)      \
    F(divide)        \
    F(remainder)     \
    F(divmod)        \
    F(power)         \
    F(square)        \
    F(reciprocal)    \
    F(_ones_like)    \
    F(sqrt)          \
    F(cbrt)          \
    F(negative)      \
    F(positive)      \
    F(absolute)      \
    F(invert)        \
    F(left_shift)    \
    F(right_shift)   \
    F(bitwise_and)   \
    F(bitwise_xor)   \
    F(bitwise_or)    \
    F(less)          \
    F(less_equal)    \
    F(equal)         \
    F(not_equal)     \
    F(greater)       \
    F(greater_equal) \
    F(floor_divide)  \
    F(true_divide)   \
    F(logical_or)    \
    F(logical_and)   \
    F(floor)         \
    F(ceil)          \
    F(maximum)       \
    F(minimum)       \
    F(rint)          \
    F(conjugate)     \
    F(matmul)        \
    F(clip)          \
    F(real)          \
    F(imag)

#define NPY_MODULE_STATE_OBJECT_FIELDS(F) \
    F(typeDict)        \
    F(current_handler) \
    F(global_pytype_to_type_dict)

/*
 * Expand a field list into the PyObject * members it names, so the structs
 * below are declared from the same list the traverse and clear functions use.
 */
#define NPY_DECLARE_ONE_PYOBJECT_FIELD(name) PyObject *name;
#define NPY_DECLARE_PYOBJECT_FIELDS(list) list(NPY_DECLARE_ONE_PYOBJECT_FIELD)

#define NPY_FIELD_COUNT_ONE(name) + 1
#define NPY_FIELD_COUNT(list) (0 list(NPY_FIELD_COUNT_ONE))

#define NPY_ERRMODE_STRING_COUNT 6

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_ */

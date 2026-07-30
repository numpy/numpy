#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_

/*
 * Single source of truth for the PyObject * members of multiarray_umath_state.
 *
 * Each macro below takes a macro F and applies it once per field name.
 * multiarray_umath_traverse() and multiarray_umath_clear() are both expanded
 * from these lists, so the two can never disagree with each other.
 *
 * The static assertions in module_state.h tie each list back to the size of
 * the struct it describes, so adding a member to one of those structs without
 * adding it here is a compile error rather than a silently missed reference.
 *
 * interned_str.errmode_strings is a fixed-size array rather than a plain
 * member, so it is not in the list below; it is handled explicitly at each use
 * site and accounted for separately in the assertion.
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
    F(partition)            \
    F(argpartition)         \
    F(_set_dtype)           \
    F(year)                 \
    F(month)                \
    F(day)                  \
    F(hour)                 \
    F(minute)               \
    F(second)               \
    F(microsecond)          \
    F(tzinfo)               \
    F(utcoffset)            \
    F(total_seconds)

#define NPY_STATIC_PYDATA_FIELDS(F) \
    F(default_truediv_type_tup)    \
    F(default_extobj_capsule)      \
    F(npy_extobj_contextvar)       \
    F(ndarray_array_ufunc)         \
    F(ndarray_array_finalize)      \
    F(ndarray_array_function)      \
    F(ndarray_set_dtype)           \
    F(ndarray_dtype_descr)         \
    F(one_obj)                     \
    F(zero_obj)                    \
    F(zero_pyint_like_arr)         \
    F(AxisError)                   \
    F(ComplexWarning)              \
    F(DTypePromotionError)         \
    F(TooHardError)                \
    F(VisibleDeprecationWarning)   \
    F(_CopyMode)                   \
    F(_NoValue)                    \
    F(_ArrayMemoryError)           \
    F(_UFuncBinaryResolutionError) \
    F(_UFuncInputCastingError)     \
    F(_UFuncNoLoopError)           \
    F(_UFuncOutputCastingError)    \
    F(math_floor_func)             \
    F(math_ceil_func)              \
    F(math_trunc_func)             \
    F(math_gcd_func)               \
    F(os_PathLike)                 \
    F(os_fspath)                   \
    F(format_options)              \
    F(legacy_resolver_promoting)   \
    F(kwnames_is_copy)             \
    F(axes_1d_obj_kwargs)          \
    F(axes_2d_obj_kwargs)          \
    F(cpu_dispatch_registry)       \
    F(VoidToGenericMethod)         \
    F(GenericToVoidMethod)         \
    F(ObjectToGenericMethod)       \
    F(GenericToObjectMethod)       \
    F(dl_call_kwnames)             \
    F(dl_cpu_device_tuple)         \
    F(dl_max_version)              \
    F(dlpack_dtype_registry)       \
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
    F(typeDict)                                 \
    F(current_handler)                          \
    F(global_pytype_to_type_dict)

/* Number of entries in a field list, e.g. NPY_FIELD_COUNT(NPY_N_OPS_FIELDS). */
#define NPY_FIELD_COUNT_ONE(name) + 1
#define NPY_FIELD_COUNT(list) (0 list(NPY_FIELD_COUNT_ONE))

/* interned_str.errmode_strings[] — see comment above. */
#define NPY_ERRMODE_STRING_COUNT 6

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_ */

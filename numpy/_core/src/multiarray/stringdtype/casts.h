#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_


#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts(void);

// Load the string representation for *ps*. This might be the packed string
// itself or, for a missing entry, str(na_object) when the descriptor has a
// non-string na_object and its default string otherwise (the na_object for a
// string na_object, empty without one).  The caller must hold *allocator*,
// acquired from *descr*.
NPY_NO_EXPORT int
load_nullable_string(const PyArray_StringDTypeObject *descr,
                     const npy_packed_static_string *ps,
                     npy_static_string *s,
                     npy_string_allocator *allocator,
                     const char *context);

// Find a fixed-width NPY_STRING, NPY_UNICODE or NPY_VOID descriptor wide
// enough to store every entry of the StringDType array *arr* without
// truncation.
// Returns NULL with an error set on failure.
NPY_NO_EXPORT PyArray_Descr *
stringdtype_find_fixed_width_descr(PyArrayObject *arr, int type_num);

// Whether obj is a NaN missing value: a float NaN or a complex value with a
// NaN component (np.isnan semantics). Returns 1 if it is, 0 if not, and -1 on
// error. Used by stringdtype_setitem; kept consistent with float_is_nan_na in
// the float-to-string casts.
NPY_NO_EXPORT int
pyobj_is_nan_na(PyObject *obj);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */

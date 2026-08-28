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

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */

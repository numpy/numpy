#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_


#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts(void);

// Load the effective value of *ps* for a conversion to a fixed-width dtype;
// the caller must hold *allocator*, acquired from *descr*.
NPY_NO_EXPORT int
load_nullable_string(const PyArray_StringDTypeObject *descr,
                     const npy_packed_static_string *ps,
                     npy_static_string *s,
                     npy_string_allocator *allocator,
                     const char *context);

// Find a fixed-width NPY_STRING or NPY_UNICODE descriptor wide enough to
// store every entry of the StringDType array *arr* without truncation.
// Returns NULL with an error set on failure.
NPY_NO_EXPORT PyArray_Descr *
stringdtype_find_fixed_width_descr(PyArrayObject *arr, int type_num);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */

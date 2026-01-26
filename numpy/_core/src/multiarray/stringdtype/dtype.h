#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

// not publicly exposed by the static string library so we need to define
// this here so we can define `elsize` and `alignment` on the descr
//
// if the layout of npy_packed_static_string ever changes in the future
// this may need to be updated.
#define SIZEOF_NPY_PACKED_STATIC_STRING 2 * sizeof(size_t)
#define ALIGNOF_NPY_PACKED_STATIC_STRING _Alignof(size_t)

NPY_NO_EXPORT PyObject *
new_stringdtype_instance(PyObject *na_object, int coerce);

NPY_NO_EXPORT int
init_string_dtype(void);

// Assumes that the caller has already acquired the allocator locks for both
// descriptors
NPY_NO_EXPORT int
_compare(void *a, void *b, PyArray_StringDTypeObject *descr_a,
         PyArray_StringDTypeObject *descr_b);

NPY_NO_EXPORT int
init_string_na_object(PyObject *mod);

NPY_NO_EXPORT int
stringdtype_setitem(PyArray_StringDTypeObject *descr, PyObject *obj, char **dataptr);

// the locks on both allocators must be acquired before calling this function
NPY_NO_EXPORT int
free_and_copy(npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator,
              const npy_packed_static_string *in,
              npy_packed_static_string *out, const char *location);

NPY_NO_EXPORT int
load_new_string(npy_packed_static_string *out, npy_static_string *out_ss,
                size_t num_bytes, npy_string_allocator *allocator,
                const char *err_context);

NPY_NO_EXPORT PyArray_Descr *
stringdtype_finalize_descr(PyArray_Descr *dtype);

NPY_NO_EXPORT int
_eq_comparison(int scoerce, int ocoerce, PyObject *sna, PyObject *ona);

NPY_NO_EXPORT int
stringdtype_compatible_na(PyObject *na1, PyObject *na2, PyObject **out_na);

NPY_NO_EXPORT int
na_eq_cmp(PyObject *a, PyObject *b);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_ */

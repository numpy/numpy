#ifndef NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_


/*
 * In some API calls we wish to allow users to pass a DType class or a
 * dtype instances with different meanings.
 * This struct is mainly used for the argument parsing in
 * `PyArray_DTypeOrDescrConverter`.
 */
typedef struct {
    PyArray_DTypeMeta *dtype;
    PyArray_Descr *descr;
} npy_dtype_info;


NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterOptional(PyObject *, npy_dtype_info *dt_info);

NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterRequired(PyObject *, npy_dtype_info *dt_info);

NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyArray_Descr *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType);

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(
        PyArray_Descr *, void *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(
        PyArray_Descr *self, void *);

/*
 * offset:    A starting offset.
 * alignment: A power-of-two alignment.
 *
 * This macro returns the smallest value >= 'offset'
 * that is divisible by 'alignment'. Because 'alignment'
 * is a power of two and integers are twos-complement,
 * it is possible to use some simple bit-fiddling to do this.
 */
#define NPY_NEXT_ALIGNED_OFFSET(offset, alignment) \
                (((offset) + (alignment) - 1) & (-(alignment)))

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj);


NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype);

/*
 * Filter the fields of a dtype to only those in the list of strings, ind.
 *
 * No type checking is performed on the input.
 *
 * Raises:
 *   ValueError - if a field is repeated
 *   KeyError - if an invalid field name (or any field title) is used
 */
NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(PyArray_Descr *self, PyObject *ind);

extern NPY_NO_EXPORT char const *_datetime_strings[];

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_ */

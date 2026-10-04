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

NPY_NO_EXPORT void
PyArray_ExtractDTypeAndDescriptor(PyArray_Descr *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType);

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(
        PyArray_Descr *, void *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(
        PyArray_Descr *self, void *);

NPY_NO_EXPORT PyObject *array_protocol_descr_get(PyArray_Descr *self);

static inline int
npy_add_to_descr_size(npy_intp *size, npy_intp increment)
{
    if (increment < 0 || *size > NPY_MAX_INTP - increment) {
        PyErr_SetString(PyExc_ValueError, "structured dtype is too large");
        return -1;
    }
    *size += increment;
    return 0;
}

/*
 * Round a descriptor size up to the next multiple of a power-of-two
 * alignment. The bit mask computes the required padding without overflowing
 * the size; npy_add_to_descr_size checks whether adding it would overflow.
 */
static inline int
npy_align_descr_size(npy_intp *size, npy_intp alignment)
{
    if (alignment <= 1) {
        return 0;
    }
    npy_intp padding = (-*size) & (alignment - 1);
    return npy_add_to_descr_size(size, padding);
}

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);


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
arraydescr_field_subset_view(_PyArray_LegacyDescr *self, PyObject *ind);

/*
 * Create a new subarray dtype from `base` and `shape`.
 */
NPY_NO_EXPORT PyArray_Descr *
arraydescr_new_from_subarray(PyArray_Descr *base, PyObject *shape);

extern NPY_NO_EXPORT char const *_datetime_strings[];

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_ */

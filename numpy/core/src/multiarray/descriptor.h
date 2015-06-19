#ifndef _NPY_ARRAYDESCR_H_
#define _NPY_ARRAYDESCR_H_

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(PyArray_Descr *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(PyArray_Descr *self);

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj);


NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype);

/*
 * Creates a string repr of the dtype, excluding the 'dtype()' part
 * surrounding the object. This object may be a string, a list, or
 * a dict depending on the nature of the dtype. This
 * is the object passed as the first parameter to the dtype
 * constructor, and if no additional constructor parameters are
 * given, will reproduce the exact memory layout.
 *
 * If 'shortrepr' is non-zero, this creates a shorter repr using
 * 'kind' and 'itemsize', instead of the longer type name.
 *
 * If 'includealignflag' is true, this includes the 'align=True' parameter
 * inside the struct dtype construction dict when needed. Use this flag
 * if you want a proper repr string without the 'dtype()' part around it.
 *
 * If 'includealignflag' is false, this does not preserve the
 * 'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
 * struct arrays like the regular repr does, because the 'align'
 * flag is not part of first dtype constructor parameter. This
 * mode is intended for a full 'repr', where the 'align=True' is
 * provided as the second parameter.
 */
NPY_NO_EXPORT PyObject *
arraydescr_construction_repr(PyArray_Descr *dtype, int includealignflag,
                                int shortrepr);

extern NPY_NO_EXPORT char *_datetime_strings[];

#endif

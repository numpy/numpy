#ifndef _NPY_ARRAYDESCR_H_
#define _NPY_ARRAYDESCR_H_

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(PyArray_Descr *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(PyArray_Descr *self);

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj);

/*
 * This creates a shorter repr using the 'kind' and 'itemsize',
 * instead of the longer type name. It also creates the input
 * for constructing a dtype rather than the full dtype function
 * call.
 *
 * This does not preserve the 'align=True' parameter
 * for structured arrays like the regular repr does.
 */
NPY_NO_EXPORT PyObject *
arraydescr_short_construction_repr(PyArray_Descr *dtype);

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT char *_datetime_strings[];
#endif

#endif

#ifndef _NPY_ARRAYDESCR_H_
#define _NPY_ARRAYDESCR_H_

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(PyArray_Descr *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(PyArray_Descr *self);

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj);

extern char **_datetime_strings;

#endif

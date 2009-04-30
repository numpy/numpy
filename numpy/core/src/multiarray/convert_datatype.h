#ifndef _NPY_ARRAY_CONVERT_DATATYPE_H_
#define _NPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *mp, PyArray_Descr *at, int fortran);

#endif

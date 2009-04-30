#ifndef _NPY_ARRAY_CONVERT_DATATYPE_H_
#define _NPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *mp, PyArray_Descr *at, int fortran);

NPY_NO_EXPORT int
PyArray_CastTo(PyArrayObject *out, PyArrayObject *mp);

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);

NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype);

NPY_NO_EXPORT Bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

#endif

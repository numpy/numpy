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

/* This can replace PyArray_CanCastTo for NumPy 2.0 (ABI change) */
NPY_NO_EXPORT npy_bool
can_cast_to(PyArray_Descr *from, PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT npy_bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

#endif

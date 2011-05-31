#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryOperationTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_AdditionTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_SubtractionTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_MultiplicationTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);
NPY_NO_EXPORT int
PyUFunc_DivisionTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

#endif

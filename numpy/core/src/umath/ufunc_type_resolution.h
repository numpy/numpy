#ifndef _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_
#define _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_SimpleUnaryOperationTypeResolution(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata);

NPY_NO_EXPORT int
PyUFunc_OnesLikeTypeResolution(PyUFuncObject *ufunc,
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
PyUFunc_AbsoluteTypeResolution(PyUFuncObject *ufunc,
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

/*
 * Does a linear search for the best inner loop of the ufunc.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
find_best_ufunc_inner_loop(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        int any_object,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata);

/*
 * Does a linear search for the inner loop of the ufunc specified by type_tup.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
find_specified_ufunc_inner_loop(PyUFuncObject *self,
                        PyObject *type_tup,
                        PyArrayObject **op,
                        NPY_CASTING casting,
                        int any_object,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata);

#endif

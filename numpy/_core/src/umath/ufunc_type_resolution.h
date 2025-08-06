#ifndef _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_
#define _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolver(PyUFuncObject *ufunc,
                                           NPY_CASTING casting,
                                           PyArrayObject **operands,
                                           PyObject *type_tup,
                                           PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_NegativeTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_OnesLikeTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_SimpleUniformOperationTypeResolver(PyUFuncObject *ufunc,
                                          NPY_CASTING casting,
                                          PyArrayObject **operands,
                                          PyObject *type_tup,
                                          PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_AbsoluteTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_IsNaTTypeResolver(PyUFuncObject *ufunc,
                          NPY_CASTING casting,
                          PyArrayObject **operands,
                          PyObject *type_tup,
                          PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_IsFiniteTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_AdditionTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_SubtractionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_MultiplicationTypeResolver(PyUFuncObject *ufunc,
                                   NPY_CASTING casting,
                                   PyArrayObject **operands,
                                   PyObject *type_tup,
                                   PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_TrueDivisionTypeResolver(PyUFuncObject *ufunc,
                                 NPY_CASTING casting,
                                 PyArrayObject **operands,
                                 PyObject *type_tup,
                                 PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_DivisionTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_RemainderTypeResolver(PyUFuncObject *ufunc,
                              NPY_CASTING casting,
                              PyArrayObject **operands,
                              PyObject *type_tup,
                              PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_DivmodTypeResolver(PyUFuncObject *ufunc,
                              NPY_CASTING casting,
                              PyArrayObject **operands,
                              PyObject *type_tup,
                              PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_ValidateOutCasting(PyUFuncObject *ufunc,
        NPY_CASTING casting, PyArrayObject **operands, PyArray_Descr **dtypes);

/*
 * Does a linear search for the best inner loop of the ufunc.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
linear_search_type_resolver(PyUFuncObject *self,
                            PyArrayObject **op,
                            NPY_CASTING input_casting,
                            NPY_CASTING output_casting,
                            int any_object,
                            PyArray_Descr **out_dtype);

/*
 * Does a linear search for the inner loop of the ufunc specified by type_tup.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
type_tuple_type_resolver(PyUFuncObject *self,
                         PyObject *type_tup,
                         PyArrayObject **op,
                         NPY_CASTING input_casting,
                         NPY_CASTING casting,
                         int any_object,
                         PyArray_Descr **out_dtype);

NPY_NO_EXPORT int
PyUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                       PyArray_Descr *const *dtypes,
                                       PyUFuncGenericFunction *out_innerloop,
                                       void **out_innerloopdata,
                                       int *out_needs_api);

NPY_NO_EXPORT int
raise_no_loop_found_error(PyUFuncObject *ufunc, PyObject **dtypes);

#ifdef __cplusplus
}
#endif

#endif

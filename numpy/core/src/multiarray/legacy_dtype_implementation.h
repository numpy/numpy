#ifndef _NPY_LEGACY_DTYPE_IMPLEMENTATION_H
#define _NPY_LEGACY_DTYPE_IMPLEMENTATION_H


NPY_NO_EXPORT unsigned char
PyArray_LegacyEquivTypes(PyArray_Descr *type1, PyArray_Descr *type2);

NPY_NO_EXPORT unsigned char
PyArray_LegacyEquivTypenums(int typenum1, int typenum2);

NPY_NO_EXPORT int
PyArray_LegacyCanCastSafely(int fromtype, int totype);

NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTo(PyArray_Descr *from, PyArray_Descr *to);

NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting);

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it returns it as-is.
 *
 * Usually, if data_obj is not an array, dtype should be the result
 * given by the PyArray_GetArrayParamsFromObject function.
 *
 * The data_obj may be NULL if just a dtype is known for the source.
 *
 * If *flex_dtype is NULL, returns immediately, without setting an
 * exception, leaving any previous error handling intact.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptFlexibleDType(PyArray_Descr *data_dtype, PyArray_Descr *flex_dtype);

#endif /*_NPY_LEGACY_DTYPE_IMPLEMENTATION_H*/

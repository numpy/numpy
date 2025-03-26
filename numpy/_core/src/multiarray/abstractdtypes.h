#ifndef NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_

#include "numpy/ndarraytypes.h"
#include "arrayobject.h"
#include "dtypemeta.h"


#ifdef __cplusplus
extern "C" {
#endif

/*
 * These are mainly needed for value based promotion in ufuncs.  It
 * may be necessary to make them (partially) public, to allow user-defined
 * dtypes to perform value based casting.
 */
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_IntAbstractDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_FloatAbstractDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_ComplexAbstractDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyLongDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyFloatDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyComplexDType;

NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes(void);


/*
 * When we get a Python int, float, or complex, we may have to use weak
 * promotion logic.
 * To implement this, we sometimes have to tag the converted (temporary)
 * array when the original object was a Python scalar.
 *
 * @param obj The original Python object.
 * @param arr The array into which the Python object was converted.
 * @param[in,out] **dtype A pointer to the array's DType, if not NULL it will be
 *        replaced with the abstract DType.
 * @return 0 if the `obj` was not a python scalar, and 1 if it was.
 */
static inline int
npy_mark_tmp_array_if_pyscalar(
        PyObject *obj, PyArrayObject *arr, PyArray_DTypeMeta **dtype)
{
    if (PyLong_CheckExact(obj)) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_INT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyLongDType);
            Py_SETREF(*dtype, &PyArray_PyLongDType);
        }
        return 1;
    }
    else if (PyFloat_CheckExact(obj)) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_FLOAT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyFloatDType);
            Py_SETREF(*dtype, &PyArray_PyFloatDType);
        }
        return 1;
    }
    else if (PyComplex_CheckExact(obj)) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_COMPLEX;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyComplexDType);
            Py_SETREF(*dtype, &PyArray_PyComplexDType);
        }
        return 1;
    }
    return 0;
}


NPY_NO_EXPORT int
npy_update_operand_for_scalar(
    PyArrayObject **operand, PyObject *scalar, PyArray_Descr *descr,
    NPY_CASTING casting);


NPY_NO_EXPORT PyArray_Descr *
npy_find_descr_for_scalar(
    PyObject *scalar, PyArray_Descr *original_descr,
    PyArray_DTypeMeta *in_DT, PyArray_DTypeMeta *op_DT);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_ */

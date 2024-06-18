#ifndef NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_

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
    /*
     * We check the array dtype for two reasons: First, booleans are
     * integer subclasses.  Second, an int, float, or complex could have
     * a custom DType registered, and then we should use that.
     * Further, `np.float64` is a double subclass, so must reject it.
     */
    // TODO,NOTE: This function should be changed to do exact long checks
    //            For NumPy 2.1!
    if (PyLong_Check(obj)
            && (PyArray_ISINTEGER(arr) || PyArray_ISOBJECT(arr))) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_INT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyLongDType);
            Py_SETREF(*dtype, &PyArray_PyLongDType);
        }
        return 1;
    }
    else if (PyFloat_Check(obj) && !PyArray_IsScalar(obj, Double)
             && PyArray_TYPE(arr) == NPY_DOUBLE) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_FLOAT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyFloatDType);
            Py_SETREF(*dtype, &PyArray_PyFloatDType);
        }
        return 1;
    }
    else if (PyComplex_Check(obj) && !PyArray_IsScalar(obj, CDouble)
             && PyArray_TYPE(arr) == NPY_CDOUBLE) {
        ((PyArrayObject_fields *)arr)->flags |= NPY_ARRAY_WAS_PYTHON_COMPLEX;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyComplexDType);
            Py_SETREF(*dtype, &PyArray_PyComplexDType);
        }
        return 1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_ */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ABSTRACTDTYPES_H_

#include "numpy/ndarraytypes.h"
#include "arrayobject.h"
#include "dtypemeta.h"


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Abstract DType classes representing the numerical "kind" hierarchy that
 * mirrors the NumPy scalar type tree (numeric -> integer -> signed/unsigned,
 * numeric -> inexact -> float/complex).  They are dynamically created during
 * module init via ``PyType_FromMetaclass`` so the storage is held in
 * pointer variables; the public ``PyArray_*AbstractDType`` names below remain
 * usable as struct lvalues (and ``&PyArray_*AbstractDType`` resolves to the
 * pointer).
 *
 * They are mainly needed for value based promotion in ufuncs and for the
 * array-API "kind" classifications used by ``isdtype`` / ``issubdtype``.
 */
NPY_NO_EXPORT extern PyArray_DTypeMeta *_NumericAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_IntegerAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_SignedIntegerAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_UnsignedIntegerAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_InexactAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_FloatAbstract_dtype;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_ComplexAbstract_dtype;

#define PyArray_NumericAbstractDType (*_NumericAbstract_dtype)
#define PyArray_IntAbstractDType (*_IntegerAbstract_dtype)
#define PyArray_SignedIntegerAbstractDType (*_SignedIntegerAbstract_dtype)
#define PyArray_UnsignedIntegerAbstractDType (*_UnsignedIntegerAbstract_dtype)
#define PyArray_InexactAbstractDType (*_InexactAbstract_dtype)
#define PyArray_FloatAbstractDType (*_FloatAbstract_dtype)
#define PyArray_ComplexAbstractDType (*_ComplexAbstract_dtype)

/*
 * The "implicit" DType classes for Python ``int``, ``float`` and ``complex``
 * literals (used by value-based promotion).  Like the abstract DTypes above,
 * they are created via ``PyType_FromMetaclass`` and stored in pointer
 * variables.  ``&PyArray_PyLongDType`` resolves to the pointer.
 */
NPY_NO_EXPORT extern PyArray_DTypeMeta *_PyLongDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_PyFloatDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta *_PyComplexDType;

#define PyArray_PyLongDType (*_PyLongDType)
#define PyArray_PyFloatDType (*_PyFloatDType)
#define PyArray_PyComplexDType (*_PyComplexDType)

/*
 * Create the abstract DType classes (NumericAbstractDType, IntegerAbstractDType,
 * SignedIntegerAbstractDType, UnsignedIntegerAbstractDType, InexactAbstractDType,
 * FloatAbstractDType, ComplexAbstractDType), expose them on ``numpy.dtypes``,
 * and set up the implicit ``PyLongDType`` / ``PyFloatDType`` / ``PyComplexDType``
 * value-based-promotion helpers.  Must be called after ``PyArrayDTypeMeta_Type``
 * is ready and before any code that uses these abstracts as ``tp_base``
 * (e.g. ``set_typeinfo``).
 */
NPY_NO_EXPORT int
initialize_abstract_dtypes(void);

/*
 * Map the Python ``str`` / ``bytes`` / ``bool`` types to the corresponding
 * NumPy DType for scalar discovery.  Must be called after ``set_typeinfo``
 * because it looks up the legacy DType classes by type number.
 */
NPY_NO_EXPORT int
map_legacy_pytypes_to_dtypes(void);


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
        _PyArray_GET_ITEM_DATA(arr)->flags |= NPY_ARRAY_WAS_PYTHON_INT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyLongDType);
            Py_SETREF(*dtype, &PyArray_PyLongDType);
        }
        return 1;
    }
    else if (PyFloat_CheckExact(obj)) {
        _PyArray_GET_ITEM_DATA(arr)->flags |= NPY_ARRAY_WAS_PYTHON_FLOAT;
        if (dtype != NULL) {
            Py_INCREF(&PyArray_PyFloatDType);
            Py_SETREF(*dtype, &PyArray_PyFloatDType);
        }
        return 1;
    }
    else if (PyComplex_CheckExact(obj)) {
        _PyArray_GET_ITEM_DATA(arr)->flags |= NPY_ARRAY_WAS_PYTHON_COMPLEX;
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

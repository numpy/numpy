#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"


static NPY_INLINE PyArray_Descr *
int_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_LONG);
}

static PyArray_Descr *
discover_descriptor_from_pyint(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyLong_Check(obj));
    /*
     * We check whether long is good enough. If not, check longlong and
     * unsigned long before falling back to `object`.
     */
    long long value = PyLong_AsLongLong(obj);
    if (error_converting(value)) {
        PyErr_Clear();
    }
    else {
        if (NPY_MIN_LONG <= value && value <= NPY_MAX_LONG) {
            return PyArray_DescrFromType(NPY_LONG);
        }
        return PyArray_DescrFromType(NPY_LONGLONG);
    }

    unsigned long long uvalue = PyLong_AsUnsignedLongLong(obj);
    if (uvalue == (unsigned long long)-1 && PyErr_Occurred()){
        PyErr_Clear();
    }
    else {
        return PyArray_DescrFromType(NPY_ULONGLONG);
    }

    return PyArray_DescrFromType(NPY_OBJECT);
}


static NPY_INLINE PyArray_Descr *
float_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_DOUBLE);
}


static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
}

static NPY_INLINE PyArray_Descr *
complex_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_CDOUBLE);
}

static PyArray_Descr*
discover_descriptor_from_pycomplex(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyComplex_CheckExact(obj));
    return PyArray_DescrFromType(NPY_COMPLEX128);
}


NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes()
{
    ((PyTypeObject *)&PyArray_PyIntAbstractDType)->tp_base = &PyArrayDescr_Type;
    PyArray_PyIntAbstractDType.scalar_type = &PyLong_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyIntAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyFloatAbstractDType)->tp_base = &PyArrayDescr_Type;
    PyArray_PyFloatAbstractDType.scalar_type = &PyFloat_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyComplexAbstractDType)->tp_base = &PyArrayDescr_Type;
    PyArray_PyComplexAbstractDType.scalar_type = &PyComplex_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyComplexAbstractDType) < 0) {
        return -1;
    }

    /* Register the new DTypes for discovery */
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyIntAbstractDType, &PyLong_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyFloatAbstractDType, &PyFloat_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyComplexAbstractDType, &PyComplex_Type, NPY_FALSE) < 0) {
        return -1;
    }

    /*
     * Map str, bytes, and bool, for which we do not need abstract versions
     * to the NumPy DTypes. This is done here using the `is_known_scalar_type`
     * function.
     * TODO: The `is_known_scalar_type` function is considered preliminary,
     *       the same could be achieved e.g. with additional abstract DTypes.
     */
    PyArray_DTypeMeta *dtype;
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_UNICODE));
    if (_PyArray_MapPyTypeToDType(dtype, &PyUnicode_Type, NPY_FALSE) < 0) {
        return -1;
    }

    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_STRING));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBytes_Type, NPY_FALSE) < 0) {
        return -1;
    }
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_BOOL));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBool_Type, NPY_FALSE) < 0) {
        return -1;
    }

    return 0;
}


/*
 * The following functions define the "common DType" for the abstract dtypes.
 *
 * Note that the logic with respect to the "higher" dtypes such as floats
 * could likely be more logically defined for them, but since NumPy dtypes
 * largely "know" each other, that is not necessary.
 */
static PyArray_DTypeMeta *
int_common_dtype(PyArray_DTypeMeta *NPY_UNUSED(cls), PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL) {
            /* Use the default integer for bools: */
            return PyArray_DTypeFromTypeNum(NPY_LONG);
        }
        else if (PyTypeNum_ISNUMBER(other->type_num) ||
                 other->type_num == NPY_TIMEDELTA) {
            /* All other numeric types (ant timedelta) are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_UINT8);
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL || PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return PyArray_DTypeFromTypeNum(NPY_DOUBLE);
        }
        else if (PyTypeNum_ISNUMBER(other->type_num)) {
            /* All other numeric types (float+complex) are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (other == &PyArray_PyIntAbstractDType) {
        Py_INCREF(cls);
        return cls;
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_HALF);
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
complex_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL ||
                PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return PyArray_DTypeFromTypeNum(NPY_CDOUBLE);
        }
        else if (PyTypeNum_ISFLOAT(other->type_num)) {
            /*
             * For floats we choose the equivalent precision complex, although
             * there is no CHALF, so half also goes to CFLOAT.
             */
            if (other->type_num == NPY_HALF || other->type_num == NPY_FLOAT) {
                return PyArray_DTypeFromTypeNum(NPY_CFLOAT);
            }
            if (other->type_num == NPY_DOUBLE) {
                return PyArray_DTypeFromTypeNum(NPY_CDOUBLE);
            }
            assert(other->type_num == NPY_LONGDOUBLE);
            return PyArray_DTypeFromTypeNum(NPY_CLONGDOUBLE);
        }
        else if (PyTypeNum_ISCOMPLEX(other->type_num)) {
            /* All other numeric types are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_CFLOAT);
    }
    else if (other == &PyArray_PyIntAbstractDType ||
             other == &PyArray_PyFloatAbstractDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


/*
 * TODO: These abstract DTypes also carry the dual role of representing
 *       `Floating`, `Complex`, and `Integer` (both signed and unsigned).
 *       They will have to be renamed and exposed in that capacity.
 */
NPY_DType_Slots pyintabstractdtype_slots = {
    .default_descr = int_default_descriptor,
    .discover_descr_from_pyobject = discover_descriptor_from_pyint,
    .common_dtype = int_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyIntAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._IntegerAbstractDType",
    },},
    .flags = NPY_DT_ABSTRACT,
    .dt_slots = &pyintabstractdtype_slots,
};


NPY_DType_Slots pyfloatabstractdtype_slots = {
    .default_descr = float_default_descriptor,
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .common_dtype = float_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._FloatAbstractDType",
    },},
    .flags = NPY_DT_ABSTRACT,
    .dt_slots = &pyfloatabstractdtype_slots,
};


NPY_DType_Slots pycomplexabstractdtype_slots = {
    .default_descr = complex_default_descriptor,
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .common_dtype = complex_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._ComplexAbstractDType",
    },},
    .flags = NPY_DT_ABSTRACT,
    .dt_slots = &pycomplexabstractdtype_slots,
};

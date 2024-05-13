#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"


static inline PyArray_Descr *
int_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_INTP);
}

static PyArray_Descr *
discover_descriptor_from_pylong(
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
        if (NPY_MIN_INTP <= value && value <= NPY_MAX_INTP) {
            return PyArray_DescrFromType(NPY_INTP);
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


static inline PyArray_Descr *
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

static inline PyArray_Descr *
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
    if (PyType_Ready((PyTypeObject *)&PyArray_IntAbstractDType) < 0) {
        return -1;
    }
    if (PyType_Ready((PyTypeObject *)&PyArray_FloatAbstractDType) < 0) {
        return -1;
    }
    if (PyType_Ready((PyTypeObject *)&PyArray_ComplexAbstractDType) < 0) {
        return -1;
    }
    /*
     * Delayed assignments to avoid "error C2099: initializer is not a constant"
     * in windows compilers.  Can hopefully be done in structs in the future.
     */
    ((PyTypeObject *)&PyArray_PyLongDType)->tp_base =
        (PyTypeObject *)&PyArray_IntAbstractDType;
    PyArray_PyLongDType.scalar_type = &PyLong_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyLongDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyFloatDType)->tp_base =
        (PyTypeObject *)&PyArray_FloatAbstractDType;
    PyArray_PyFloatDType.scalar_type = &PyFloat_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyFloatDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyComplexDType)->tp_base =
        (PyTypeObject *)&PyArray_ComplexAbstractDType;
    PyArray_PyComplexDType.scalar_type = &PyComplex_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyComplexDType) < 0) {
        return -1;
    }

    /* Register the new DTypes for discovery */
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyLongDType, &PyLong_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyFloatDType, &PyFloat_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyComplexDType, &PyComplex_Type, NPY_FALSE) < 0) {
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
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL) {
            /* Use the default integer for bools: */
            return NPY_DT_NewRef(&PyArray_IntpDType);
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *uint8_dt = &PyArray_UInt8DType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, uint8_dt);
        Py_DECREF(uint8_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Try again with `int8`, an error may have been set, though */
        PyArray_DTypeMeta *int8_dt = &PyArray_Int8DType;
        res = NPY_DT_CALL_common_dtype(other, int8_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* And finally, we will try the default integer, just for sports... */
        PyArray_DTypeMeta *default_int = &PyArray_IntpDType;
        res = NPY_DT_CALL_common_dtype(other, default_int);
        if (res == NULL) {
            PyErr_Clear();
        }
        return res;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL || PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return NPY_DT_NewRef(&PyArray_DoubleDType);
        }
    }
    else if (other == &PyArray_PyLongDType) {
        Py_INCREF(cls);
        return cls;
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *half_dt = &PyArray_HalfDType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, half_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Retry with double (the default float) */
        PyArray_DTypeMeta *double_dt = &PyArray_DoubleDType;
        res = NPY_DT_CALL_common_dtype(other, double_dt);
        return res;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
complex_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL ||
                PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return NPY_DT_NewRef(&PyArray_CDoubleDType);
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *cfloat_dt = &PyArray_CFloatDType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, cfloat_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Retry with cdouble (the default complex) */
        PyArray_DTypeMeta *cdouble_dt = &PyArray_CDoubleDType;
        res = NPY_DT_CALL_common_dtype(other, cdouble_dt);
        return res;

    }
    else if (other == &PyArray_PyLongDType ||
             other == &PyArray_PyFloatDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


/*
 * Define abstract numerical DTypes that all regular ones can inherit from
 * (in arraytypes.c.src).
 * Here, also define types corresponding to the python scalars.
 */
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_IntAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._IntegerAbstractDType",
        .tp_base = &PyArrayDescr_Type,
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .flags = NPY_DT_ABSTRACT,
};

NPY_DType_Slots pylongdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pylong,
    .default_descr = int_default_descriptor,
    .common_dtype = int_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyLongDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyLongDType",
        .tp_base = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .dt_slots = &pylongdtype_slots,
    .scalar_type = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_FloatAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._FloatAbstractDType",
        .tp_base = &PyArrayDescr_Type,
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .flags = NPY_DT_ABSTRACT,
};

NPY_DType_Slots pyfloatdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .default_descr = float_default_descriptor,
    .common_dtype = float_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyFloatDType",
        .tp_base = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .dt_slots = &pyfloatdtype_slots,
    .scalar_type = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_ComplexAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._ComplexAbstractDType",
        .tp_base = &PyArrayDescr_Type,
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .flags = NPY_DT_ABSTRACT,
};

NPY_DType_Slots pycomplexdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .default_descr = complex_default_descriptor,
    .common_dtype = complex_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyComplexDType",
        .tp_base = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .dt_slots = &pycomplexdtype_slots,
    .scalar_type = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
};

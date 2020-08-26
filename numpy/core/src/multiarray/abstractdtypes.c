#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"


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


static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
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
    PyArrayAbstractObjDTypeMeta_Type.tp_base = &PyArrayDTypeMeta_Type;
    if (PyType_Ready(&PyArrayAbstractObjDTypeMeta_Type) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyIntAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
    PyArray_PyIntAbstractDType.scalar_type = &PyLong_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyIntAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyFloatAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
    PyArray_PyFloatAbstractDType.scalar_type = &PyFloat_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyComplexAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
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



/* Note: This is currently largely not used, but will be required eventually. */
NPY_NO_EXPORT PyTypeObject PyArrayAbstractObjDTypeMeta_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numpy._AbstractObjDTypeMeta",
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_doc = "Helper MetaClass for value based casting AbstractDTypes.",
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyIntAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy._PyIntBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pyint,
    .kind = 'i',
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy._PyFloatBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .kind = 'f',
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy._PyComplexBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .kind = 'c',
};


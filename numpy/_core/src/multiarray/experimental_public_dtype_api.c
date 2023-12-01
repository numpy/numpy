#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "common.h"

#include "experimental_public_dtype_api.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "array_coercion.h"
#include "convert_datatype.h"
#include "common_dtype.h"
#include "umathmodule.h"
#include "abstractdtypes.h"


int
PyArrayInitDTypeMeta_FromSpec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec)
{
    if (!PyObject_TypeCheck(DType, &PyArrayDTypeMeta_Type)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Passed in DType must be a valid (initialized) DTypeMeta "
                "instance!");
        return -1;
    }

    if (((PyTypeObject *)DType)->tp_repr == PyArrayDescr_Type.tp_repr
            || ((PyTypeObject *)DType)->tp_str == PyArrayDescr_Type.tp_str) {
        PyErr_SetString(PyExc_TypeError,
                "A custom DType must implement `__repr__` and `__str__` since "
                "the default inherited version (currently) fails.");
        return -1;
    }

    if (spec->typeobj == NULL || !PyType_Check(spec->typeobj)) {
        PyErr_SetString(PyExc_TypeError,
                "Not giving a type object is currently not supported, but "
                "is expected to be supported eventually.  This would mean "
                "that e.g. indexing a NumPy array will return a 0-D array "
                "and not a scalar.");
        return -1;
    }

    /* Check and handle flags: */
    int allowed_flags = NPY_DT_PARAMETRIC | NPY_DT_ABSTRACT | NPY_DT_NUMERIC;
    if (spec->flags & ~(allowed_flags)) {
        PyErr_SetString(PyExc_RuntimeError,
                "invalid DType flags specified, only NPY_DT_PARAMETRIC, "
                "NPY_DT_ABSTRACT, and NPY_DT_NUMERIC are valid flags for "
                "user DTypes.");
        return -1;
    }

    if (spec->casts == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "DType must at least provide a function to cast (or just copy) "
            "between its own instances!");
        return -1;
    }

    dtypemeta_initialize_struct_from_spec(DType, spec);

    if (NPY_DT_SLOTS(DType)->setitem == NULL
            || NPY_DT_SLOTS(DType)->getitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "A DType must provide a getitem/setitem (there may be an "
                "exception here in the future if no scalar type is provided)");
        return -1;
    }

    if (NPY_DT_SLOTS(DType)->ensure_canonical == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "A DType must provide an ensure_canonical implementation.");
        return -1;
    }

    /*
     * Now that the spec is read we can check that all required functions were
     * defined by the user.
     */
    if (spec->flags & NPY_DT_PARAMETRIC) {
        if (NPY_DT_SLOTS(DType)->common_instance == NULL ||
                NPY_DT_SLOTS(DType)->discover_descr_from_pyobject
                        == &dtypemeta_discover_as_default) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Parametric DType must define a common-instance and "
                    "descriptor discovery function!");
            return -1;
        }
    }

    if (NPY_DT_SLOTS(DType)->within_dtype_castingimpl == NULL) {
        /*
         * We expect this for now. We should have a default for DType that
         * only supports simple copy (and possibly byte-order assuming that
         * they swap the full itemsize).
         */
        PyErr_SetString(PyExc_RuntimeError,
                "DType must provide a function to cast (or just copy) between "
                "its own instances!");
        return -1;
    }

    /* And finally, we have to register all the casts! */
    return 0;
}


/* Functions defined in umath/dispatching.c (same/one compilation unit) */
NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);

NPY_NO_EXPORT int
PyUFunc_AddLoopFromSpec(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);


/*
 * Function is defined in umath/wrapping_array_method.c
 * (same/one compilation unit)
 */
NPY_NO_EXPORT int
PyUFunc_AddWrappingLoop(PyObject *ufunc_obj,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
        translate_given_descrs_func *translate_given_descrs,
        translate_loop_descrs_func *translate_loop_descrs);


static int
PyUFunc_AddPromoter(
        PyObject *ufunc, PyObject *DType_tuple, PyObject *promoter)
{
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc object passed is not a ufunc!");
        return -1;
    }
    if (!PyCapsule_CheckExact(promoter)) {
        PyErr_SetString(PyExc_TypeError,
                "promoter must (currently) be a PyCapsule.");
        return -1;
    }
    if (PyCapsule_GetPointer(promoter, "numpy._ufunc_promoter") == NULL) {
        return -1;
    }
    PyObject *info = PyTuple_Pack(2, DType_tuple, promoter);
    if (info == NULL) {
        return -1;
    }
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
}


/*
 * Lightweight function fetch a default instance of a DType class.
 * Note that this version is named `_PyArray_GetDefaultDescr` with an
 * underscore.  The `singleton` slot is public, so an inline version is
 * provided that checks `singleton != NULL` first.
 */
static PyArray_Descr *
_PyArray_GetDefaultDescr(PyArray_DTypeMeta *DType)
{
    return NPY_DT_CALL_default_descr(DType);
}


NPY_NO_EXPORT PyObject *
_get_experimental_dtype_api(PyObject *NPY_UNUSED(mod), PyObject *arg)
{
    static void *experimental_api_table[47] = {
            &PyUFunc_AddLoopFromSpec,
            &PyUFunc_AddPromoter,
            &PyArrayDTypeMeta_Type,
            &PyArrayInitDTypeMeta_FromSpec,
            &PyArray_CommonDType,
            &PyArray_PromoteDTypeSequence,
            &_PyArray_GetDefaultDescr,
            &PyUFunc_AddWrappingLoop,
            &PyUFunc_GiveFloatingpointErrors,
            NULL,
            /* NumPy's builtin DTypes (starting at offset 10 going to 41) */
    };
    if (experimental_api_table[10] == NULL) {
        experimental_api_table[10] = PyArray_DTypeFromTypeNum(NPY_BOOL);
        /* Integers */
        experimental_api_table[11] = PyArray_DTypeFromTypeNum(NPY_BYTE);
        experimental_api_table[12] = PyArray_DTypeFromTypeNum(NPY_UBYTE);
        experimental_api_table[13] = PyArray_DTypeFromTypeNum(NPY_SHORT);
        experimental_api_table[14] = PyArray_DTypeFromTypeNum(NPY_USHORT);
        experimental_api_table[15] = PyArray_DTypeFromTypeNum(NPY_INT);
        experimental_api_table[16] = PyArray_DTypeFromTypeNum(NPY_UINT);
        experimental_api_table[17] = PyArray_DTypeFromTypeNum(NPY_LONG);
        experimental_api_table[18] = PyArray_DTypeFromTypeNum(NPY_ULONG);
        experimental_api_table[19] = PyArray_DTypeFromTypeNum(NPY_LONGLONG);
        experimental_api_table[20] = PyArray_DTypeFromTypeNum(NPY_ULONGLONG);
        /* Integer aliases */
        experimental_api_table[21] = PyArray_DTypeFromTypeNum(NPY_INT8);
        experimental_api_table[22] = PyArray_DTypeFromTypeNum(NPY_UINT8);
        experimental_api_table[23] = PyArray_DTypeFromTypeNum(NPY_INT16);
        experimental_api_table[24] = PyArray_DTypeFromTypeNum(NPY_UINT16);
        experimental_api_table[25] = PyArray_DTypeFromTypeNum(NPY_INT32);
        experimental_api_table[26] = PyArray_DTypeFromTypeNum(NPY_UINT32);
        experimental_api_table[27] = PyArray_DTypeFromTypeNum(NPY_INT64);
        experimental_api_table[28] = PyArray_DTypeFromTypeNum(NPY_UINT64);
        experimental_api_table[29] = PyArray_DTypeFromTypeNum(NPY_INTP);
        experimental_api_table[30] = PyArray_DTypeFromTypeNum(NPY_UINTP);
        /* Floats */
        experimental_api_table[31] = PyArray_DTypeFromTypeNum(NPY_HALF);
        experimental_api_table[32] = PyArray_DTypeFromTypeNum(NPY_FLOAT);
        experimental_api_table[33] = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
        experimental_api_table[34] = PyArray_DTypeFromTypeNum(NPY_LONGDOUBLE);
        /* Complex */
        experimental_api_table[35] = PyArray_DTypeFromTypeNum(NPY_CFLOAT);
        experimental_api_table[36] = PyArray_DTypeFromTypeNum(NPY_CDOUBLE);
        experimental_api_table[37] = PyArray_DTypeFromTypeNum(NPY_CLONGDOUBLE);
        /* String/Bytes */
        experimental_api_table[38] = PyArray_DTypeFromTypeNum(NPY_STRING);
        experimental_api_table[39] = PyArray_DTypeFromTypeNum(NPY_UNICODE);
        /* Datetime/Timedelta */
        experimental_api_table[40] = PyArray_DTypeFromTypeNum(NPY_DATETIME);
        experimental_api_table[41] = PyArray_DTypeFromTypeNum(NPY_TIMEDELTA);
        /* Object and Structured */
        experimental_api_table[42] = PyArray_DTypeFromTypeNum(NPY_OBJECT);
        experimental_api_table[43] = PyArray_DTypeFromTypeNum(NPY_VOID);
        /* Abstract */
        experimental_api_table[44] = &PyArray_PyIntAbstractDType;
        experimental_api_table[45] = &PyArray_PyFloatAbstractDType;
        experimental_api_table[46] = &PyArray_PyComplexAbstractDType;
    }

    char *env = getenv("NUMPY_EXPERIMENTAL_DTYPE_API");
    if (env == NULL || strcmp(env, "1") != 0) {
        PyErr_Format(PyExc_RuntimeError,
                "The new DType API is currently in an exploratory phase and "
                "should NOT be used for production code.  "
                "Expect modifications and crashes!  "
                "To experiment with the new API you must set "
                "`NUMPY_EXPERIMENTAL_DTYPE_API=1` as an environment variable.");
        return NULL;
    }

    long version = PyLong_AsLong(arg);
    if (error_converting(version)) {
        return NULL;
    }
    if (version != __EXPERIMENTAL_DTYPE_API_VERSION) {
        PyErr_Format(PyExc_RuntimeError,
                "Experimental DType API version %d requested, but NumPy "
                "is exporting version %d.  Recompile your DType and/or upgrade "
                "NumPy to match.",
                version, __EXPERIMENTAL_DTYPE_API_VERSION);
        return NULL;
    }

    return PyCapsule_New(&experimental_api_table,
            "experimental_dtype_api_table", NULL);
}

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
#include "dispatching.h"

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


/*
 * Function is defined in umath/wrapping_array_method.c
 * (same/one compilation unit)
 */
NPY_NO_EXPORT int
PyUFunc_AddWrappingLoop(PyObject *ufunc_obj,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
        translate_given_descrs_func *translate_given_descrs,
        translate_loop_descrs_func *translate_loop_descrs);


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
    static void *experimental_api_table[48] = {
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
        experimental_api_table[10] = &PyArray_BoolDType;
        /* Integers */
        experimental_api_table[11] = &PyArray_ByteDType;
        experimental_api_table[12] = &PyArray_UByteDType;
        experimental_api_table[13] = &PyArray_ShortDType;
        experimental_api_table[14] = &PyArray_UShortDType;
        experimental_api_table[15] = &PyArray_IntDType;
        experimental_api_table[16] = &PyArray_UIntDType;
        experimental_api_table[17] = &PyArray_LongDType;
        experimental_api_table[18] = &PyArray_ULongDType;
        experimental_api_table[19] = &PyArray_LongLongDType;
        experimental_api_table[20] = &PyArray_ULongLongDType;
        /* Integer aliases */
        experimental_api_table[21] = &PyArray_Int8DType;
        experimental_api_table[22] = &PyArray_UInt8DType;
        experimental_api_table[23] = &PyArray_Int16DType;
        experimental_api_table[24] = &PyArray_UInt16DType;
        experimental_api_table[25] = &PyArray_Int32DType;
        experimental_api_table[26] = &PyArray_UInt32DType;
        experimental_api_table[27] = &PyArray_Int64DType;
        experimental_api_table[28] = &PyArray_UInt64DType;
        experimental_api_table[29] = &PyArray_IntpDType;
        experimental_api_table[30] = &PyArray_UIntpDType;
        /* Floats */
        experimental_api_table[31] = &PyArray_HalfDType;
        experimental_api_table[32] = &PyArray_FloatDType;
        experimental_api_table[33] = &PyArray_DoubleDType;
        experimental_api_table[34] = &PyArray_LongDoubleDType;
        /* Complex */
        experimental_api_table[35] = &PyArray_CFloatDType;
        experimental_api_table[36] = &PyArray_CDoubleDType;
        experimental_api_table[37] = &PyArray_CLongDoubleDType;
        /* String/Bytes */
        experimental_api_table[38] = &PyArray_BytesDType;
        experimental_api_table[39] = &PyArray_UnicodeDType;
        /* Datetime/Timedelta */
        experimental_api_table[40] = &PyArray_DatetimeDType;
        experimental_api_table[41] = &PyArray_TimedeltaDType;
        /* Object and Structured */
        experimental_api_table[42] = &PyArray_ObjectDType;
        experimental_api_table[43] = &PyArray_VoidDType;
        /* Abstract */
        experimental_api_table[44] = &PyArray_PyIntAbstractDType;
        experimental_api_table[45] = &PyArray_PyFloatAbstractDType;
        experimental_api_table[46] = &PyArray_PyComplexAbstractDType;
        experimental_api_table[47] = &PyArray_DefaultIntDType;
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

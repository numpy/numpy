#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "common.h"

#include "public_dtype_api.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "array_coercion.h"
#include "convert_datatype.h"
#include "umathmodule.h"
#include "abstractdtypes.h"
#include "dispatching.h"

/*NUMPY_API
 *
 * Initialize a new DType.  It must currently be a static Python C type that
 * is declared as `PyArray_DTypeMeta` and not `PyTypeObject`.  Further, it
 * must subclass `np.dtype` and set its type to `PyArrayDTypeMeta_Type`
 * (before calling `PyType_Ready()`). The DTypeMeta object has additional
 * fields compared to a normal PyTypeObject!
 */
int
PyArrayInitDTypeMeta_FromSpec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec)
{
    if (!PyObject_TypeCheck(DType, PyArrayDTypeMeta_TypePtr)) {
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

    if (dtypemeta_initialize_struct_from_spec(DType, spec, 0) < 0) {
        return -1;
    }

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


void
_fill_dtype_api(void *full_api_table[])
{
    void **api_table = full_api_table + 320;

    /* The type of the DType metaclass */
    api_table[0] = PyArrayDTypeMeta_TypePtr;
    /* Boolean */
    api_table[1] = PyArray_BoolDTypePtr;
    /* Integers */
    api_table[2] = PyArray_ByteDTypePtr;
    api_table[3] = PyArray_UByteDTypePtr;
    api_table[4] = PyArray_ShortDTypePtr;
    api_table[5] = PyArray_UShortDTypePtr;
    api_table[6] = PyArray_IntDTypePtr;
    api_table[7] = PyArray_UIntDTypePtr;
    api_table[8] = PyArray_LongDTypePtr;
    api_table[9] = PyArray_ULongDTypePtr;
    api_table[10] = PyArray_LongLongDTypePtr;
    api_table[11] = PyArray_ULongLongDTypePtr;
    /* Integer aliases */
    api_table[12] = PyArray_Int8DTypePtr;
    api_table[13] = PyArray_UInt8DTypePtr;
    api_table[14] = PyArray_Int16DTypePtr;
    api_table[15] = PyArray_UInt16DTypePtr;
    api_table[16] = PyArray_Int32DTypePtr;
    api_table[17] = PyArray_UInt32DTypePtr;
    api_table[18] = PyArray_Int64DTypePtr;
    api_table[19] = PyArray_UInt64DTypePtr;
    api_table[20] = PyArray_IntpDTypePtr;
    api_table[21] = PyArray_UIntpDTypePtr;
    /* Floats */
    api_table[22] = PyArray_HalfDTypePtr;
    api_table[23] = PyArray_FloatDTypePtr;
    api_table[24] = PyArray_DoubleDTypePtr;
    api_table[25] = PyArray_LongDoubleDTypePtr;
    /* Complex */
    api_table[26] = PyArray_CFloatDTypePtr;
    api_table[27] = PyArray_CDoubleDTypePtr;
    api_table[28] = PyArray_CLongDoubleDTypePtr;
    /* String/Bytes */
    api_table[29] = PyArray_BytesDTypePtr;
    api_table[30] = PyArray_UnicodeDTypePtr;
    /* Datetime/Timedelta */
    api_table[31] = PyArray_DatetimeDTypePtr;
    api_table[32] = PyArray_TimedeltaDTypePtr;
    /* Object and Structured */
    api_table[33] = PyArray_ObjectDTypePtr;
    api_table[34] = PyArray_VoidDTypePtr;
    /* Abstract */
    api_table[35] = PyArray_PyLongDTypePtr;
    api_table[36] = PyArray_PyFloatDTypePtr;
    api_table[37] = PyArray_PyComplexDTypePtr;
    api_table[38] = PyArray_DefaultIntDTypePtr;
    /* Non-legacy DTypes that are built in to NumPy */
    api_table[39] = PyArray_StringDTypePtr;

    /* Abstract ones added directly: */
    full_api_table[366] = PyArray_IntAbstractDTypePtr;
    full_api_table[367] = PyArray_FloatAbstractDTypePtr;
    full_api_table[368] = PyArray_ComplexAbstractDTypePtr;
}

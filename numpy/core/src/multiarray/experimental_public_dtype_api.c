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


static PyArray_DTypeMeta *
dtype_does_not_promote(
        PyArray_DTypeMeta *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /* `other` is guaranteed not to be `self`, so we don't have to do much... */
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_Descr *
discover_as_default(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return NPY_DT_CALL_default_descr(cls);
}


static PyArray_Descr *
use_new_as_default(PyArray_DTypeMeta *self)
{
    PyObject *res = PyObject_CallObject((PyObject *)self, NULL);
    if (res == NULL) {
        return NULL;
    }
    /*
     * Lets not trust that the DType is implemented correctly
     * TODO: Should probably do an exact type-check (at least unless this is
     *       an abstract DType).
     */
    if (!PyArray_DescrCheck(res)) {
        PyErr_Format(PyExc_RuntimeError,
                "Instantiating %S did not return a dtype instance, this is "
                "invalid (especially without a custom `default_descr()`).",
                self);
        Py_DECREF(res);
        return NULL;
    }
    PyArray_Descr *descr = (PyArray_Descr *)res;
    /*
     * Should probably do some more sanity checks here on the descriptor
     * to ensure the user is not being naughty. But in the end, we have
     * only limited control anyway.
     */
    return descr;
}


static int
legacy_setitem_using_DType(PyObject *obj, void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return -1;
    }
    setitemfunction *setitem;
    setitem = NPY_DT_SLOTS(NPY_DTYPE(PyArray_DESCR(arr)))->setitem;
    return setitem(PyArray_DESCR(arr), obj, data);
}


static PyObject *
legacy_getitem_using_DType(void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return NULL;
    }
    getitemfunction *getitem;
    getitem = NPY_DT_SLOTS(NPY_DTYPE(PyArray_DESCR(arr)))->getitem;
    return getitem(PyArray_DESCR(arr), data);
}


/*
 * The descr->f structure used user-DTypes.  Some functions may be filled
 * from the user in the future and more could get defaults for compatibility.
 */
PyArray_ArrFuncs default_funcs = {
        .getitem = &legacy_getitem_using_DType,
        .setitem = &legacy_setitem_using_DType,
};


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

    if (DType->dt_slots != NULL) {
        PyErr_Format(PyExc_RuntimeError,
                "DType %R appears already registered?", DType);
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

    DType->flags = spec->flags;
    DType->dt_slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
    if (DType->dt_slots == NULL) {
        return -1;
    }

    /* Set default values (where applicable) */
    NPY_DT_SLOTS(DType)->discover_descr_from_pyobject = &discover_as_default;
    NPY_DT_SLOTS(DType)->is_known_scalar_type = (
            &python_builtins_are_known_scalar_types);
    NPY_DT_SLOTS(DType)->default_descr = use_new_as_default;
    NPY_DT_SLOTS(DType)->common_dtype = dtype_does_not_promote;
    /* May need a default for non-parametric? */
    NPY_DT_SLOTS(DType)->common_instance = NULL;
    NPY_DT_SLOTS(DType)->setitem = NULL;
    NPY_DT_SLOTS(DType)->getitem = NULL;
    NPY_DT_SLOTS(DType)->get_clear_loop = NULL;
    NPY_DT_SLOTS(DType)->f = default_funcs;

    PyType_Slot *spec_slot = spec->slots;
    while (1) {
        int slot = spec_slot->slot;
        void *pfunc = spec_slot->pfunc;
        spec_slot++;
        if (slot == 0) {
            break;
        }
        if ((slot < 0) ||
            ((slot > NPY_NUM_DTYPE_SLOTS) &&
             (slot <= _NPY_DT_ARRFUNCS_OFFSET)) ||
            (slot > NPY_DT_MAX_ARRFUNCS_SLOT)) {
            PyErr_Format(PyExc_RuntimeError,
                    "Invalid slot with value %d passed in.", slot);
            return -1;
        }
        /*
         * It is up to the user to get this right, the slots in the public API
         * are sorted exactly like they are stored in the NPY_DT_Slots struct
         * right now:
         */
        if (slot <= NPY_NUM_DTYPE_SLOTS) {
            // slot > NPY_NUM_DTYPE_SLOTS are PyArray_ArrFuncs
            void **current = (void **)(&(
                    NPY_DT_SLOTS(DType)->discover_descr_from_pyobject));
            current += slot - 1;
            *current = pfunc;
        }
        else {
            int f_slot = slot - _NPY_DT_ARRFUNCS_OFFSET;
            if (1 <= f_slot && f_slot <= NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS) {
                switch (f_slot) {
                    case 1:
                        NPY_DT_SLOTS(DType)->f.getitem = pfunc;
                        break;
                    case 2:
                        NPY_DT_SLOTS(DType)->f.setitem = pfunc;
                        break;
                    case 3:
                        NPY_DT_SLOTS(DType)->f.copyswapn = pfunc;
                        break;
                    case 4:
                        NPY_DT_SLOTS(DType)->f.copyswap = pfunc;
                        break;
                    case 5:
                        NPY_DT_SLOTS(DType)->f.compare = pfunc;
                        break;
                    case 6:
                        NPY_DT_SLOTS(DType)->f.argmax = pfunc;
                        break;
                    case 7:
                        NPY_DT_SLOTS(DType)->f.dotfunc = pfunc;
                        break;
                    case 8:
                        NPY_DT_SLOTS(DType)->f.scanfunc = pfunc;
                        break;
                    case 9:
                        NPY_DT_SLOTS(DType)->f.fromstr = pfunc;
                        break;
                    case 10:
                        NPY_DT_SLOTS(DType)->f.nonzero = pfunc;
                        break;
                    case 11:
                        NPY_DT_SLOTS(DType)->f.fill = pfunc;
                        break;
                    case 12:
                        NPY_DT_SLOTS(DType)->f.fillwithscalar = pfunc;
                        break;
                    case 13:
                        *NPY_DT_SLOTS(DType)->f.sort = pfunc;
                        break;
                    case 14:
                        *NPY_DT_SLOTS(DType)->f.argsort = pfunc;
                        break;
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                        PyErr_Format(
                            PyExc_RuntimeError,
                            "PyArray_ArrFunc casting slot with value %d is disabled.",
                            f_slot
                        );
                        return -1;
                    case 22:
                        NPY_DT_SLOTS(DType)->f.argmin = pfunc;
                        break;
                }
            } else {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Invalid PyArray_ArrFunc slot with value %d passed in.",
                        f_slot
                    );
                    return -1;
            }
        }
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
                        == &discover_as_default) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Parametric DType must define a common-instance and "
                    "descriptor discovery function!");
            return -1;
        }
    }

    /* invalid type num. Ideally, we get away with it! */
    DType->type_num = -1;

    /*
     * Handle the scalar type mapping.
     */
    Py_INCREF(spec->typeobj);
    DType->scalar_type = spec->typeobj;
    if (PyType_GetFlags(spec->typeobj) & Py_TPFLAGS_HEAPTYPE) {
        if (PyObject_SetAttrString((PyObject *)DType->scalar_type,
                "__associated_array_dtype__", (PyObject *)DType) < 0) {
            Py_DECREF(DType);
            return -1;
        }
    }
    if (_PyArray_MapPyTypeToDType(DType, DType->scalar_type, 0) < 0) {
        Py_DECREF(DType);
        return -1;
    }

    /* Ensure cast dict is defined (not sure we have to do it here) */
    NPY_DT_SLOTS(DType)->castingimpls = PyDict_New();
    if (NPY_DT_SLOTS(DType)->castingimpls == NULL) {
        return -1;
    }
    /*
     * And now, register all the casts that are currently defined!
     */
    if (spec->casts == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "DType must at least provide a function to cast (or just copy) "
            "between its own instances!");
        return -1;
    }

    PyArrayMethod_Spec **next_meth_spec = spec->casts;
    while (1) {
        PyArrayMethod_Spec *meth_spec = *next_meth_spec;
        next_meth_spec++;
        if (meth_spec == NULL) {
            break;
        }
        /*
         * The user doesn't know the name of DType yet, so we have to fill it
         * in for them!
         */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == NULL) {
                meth_spec->dtypes[i] = DType;
            }
        }
        /* Register the cast! */
        int res = PyArray_AddCastingImplementation_FromSpec(meth_spec, 0);

        /* Also clean up again, so nobody can get bad ideas... */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == DType) {
                meth_spec->dtypes[i] = NULL;
            }
        }

        if (res < 0) {
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
    static void *experimental_api_table[44] = {
            &PyUFunc_AddLoopFromSpec,
            &PyUFunc_AddPromoter,
            &PyArrayDTypeMeta_Type,
            &PyArrayInitDTypeMeta_FromSpec,
            &PyArray_CommonDType,
            &PyArray_PromoteDTypeSequence,
            &_PyArray_GetDefaultDescr,
            &PyUFunc_AddWrappingLoop,
            NULL,
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

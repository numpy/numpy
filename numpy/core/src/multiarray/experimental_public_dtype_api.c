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


#define EXPERIMENTAL_DTYPE_API_VERSION 2


typedef struct{
    PyTypeObject *typeobj;    /* type of python scalar or NULL */
    int flags;                /* flags, including parametric and abstract */
    /* NULL terminated cast definitions. Use NULL for the newly created DType */
    PyArrayMethod_Spec **casts;
    PyType_Slot *slots;
} PyArrayDTypeMeta_Spec;



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
        .setitem = &legacy_setitem_using_DType,
        .getitem = &legacy_getitem_using_DType
};


/* other slots are in order, so keep only last around: */
#define NUM_DTYPE_SLOTS 7


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
    if (spec->flags & ~(NPY_DT_PARAMETRIC|NPY_DT_ABSTRACT)) {
        PyErr_SetString(PyExc_RuntimeError,
                "invalid DType flags specified, only parametric and abstract "
                "are valid flags for user DTypes.");
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

    PyType_Slot *spec_slot = spec->slots;
    while (1) {
        int slot = spec_slot->slot;
        void *pfunc = spec_slot->pfunc;
        spec_slot++;
        if (slot == 0) {
            break;
        }
        if (slot > NUM_DTYPE_SLOTS || slot < 0) {
            PyErr_Format(PyExc_RuntimeError,
                    "Invalid slot with value %d passed in.", slot);
            return -1;
        }
        /*
         * It is up to the user to get this right, and slots are sorted
         * exactly like they are stored right now:
         */
        void **current = (void **)(&(
                NPY_DT_SLOTS(DType)->discover_descr_from_pyobject));
        current += slot - 1;
        *current = pfunc;
    }
    if (NPY_DT_SLOTS(DType)->setitem == NULL
            || NPY_DT_SLOTS(DType)->getitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "A DType must provide a getitem/setitem (there may be an "
                "exception here in the future if no scalar type is provided)");
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
    NPY_DT_SLOTS(DType)->f = default_funcs;
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


/* Function is defined in umath/dispatching.c (same/one compilation unit) */
NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);

static int
PyUFunc_AddLoopFromSpec(PyObject *ufunc, PyArrayMethod_Spec *spec)
{
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc object passed is not a ufunc!");
        return -1;
    }
    PyBoundArrayMethodObject *bmeth =
            (PyBoundArrayMethodObject *)PyArrayMethod_FromSpec(spec);
    if (bmeth == NULL) {
        return -1;
    }
    int nargs = bmeth->method->nin + bmeth->method->nout;
    PyObject *dtypes = PyArray_TupleFromItems(
            nargs, (PyObject **)bmeth->dtypes, 1);
    if (dtypes == NULL) {
        return -1;
    }
    PyObject *info = PyTuple_Pack(2, dtypes, bmeth->method);
    Py_DECREF(bmeth);
    Py_DECREF(dtypes);
    if (info == NULL) {
        return -1;
    }
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
}


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


NPY_NO_EXPORT PyObject *
_get_experimental_dtype_api(PyObject *NPY_UNUSED(mod), PyObject *arg)
{
    static void *experimental_api_table[] = {
            &PyUFunc_AddLoopFromSpec,
            &PyUFunc_AddPromoter,
            &PyArrayDTypeMeta_Type,
            &PyArrayInitDTypeMeta_FromSpec,
            &PyArray_CommonDType,
            &PyArray_PromoteDTypeSequence,
            NULL,
    };

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
    if (version != EXPERIMENTAL_DTYPE_API_VERSION) {
        PyErr_Format(PyExc_RuntimeError,
                "Experimental DType API version %d requested, but NumPy "
                "is exporting version %d.  Recompile your DType and/or upgrade "
                "NumPy to match.",
                version, EXPERIMENTAL_DTYPE_API_VERSION);
        return NULL;
    }

    return PyCapsule_New(&experimental_api_table,
            "experimental_dtype_api_table", NULL);
}

/*
 * This file defines an _array_converter object used internally in NumPy to
 * deal with `__array_wrap__` and `result_type()` for multiple arguments
 * where converting inputs to arrays would lose the necessary information.
 *
 * The helper thus replaces many asanyarray/asarray calls.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "array_converter.h"
#include "arraywrap.h"
#include "numpy/arrayscalars.h"
#include "npy_argparse.h"
#include "abstractdtypes.h"
#include "convert_datatype.h"
#include "descriptor.h"
#include "npy_static_data.h"
#include "ctors.h"

#include "npy_config.h"


#include "array_assign.h"

#include "common.h"
#include "get_attr_string.h"



static PyObject *
array_converter_new(
        PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    if (kwds != NULL && PyDict_GET_SIZE(kwds) != 0) {
        PyErr_SetString(PyExc_TypeError,
                "Array creation helper doesn't support keywords.");
        return NULL;
    }

    Py_ssize_t narrs_ssize_t = (args == NULL) ? 0 : PyTuple_GET_SIZE(args);
    int narrs = (int)narrs_ssize_t;
    /* Limit to NPY_MAXARGS for now. */
    if (narrs_ssize_t > NPY_MAXARGS) {
        PyErr_SetString(PyExc_RuntimeError,
            "too many arrays.");
        return NULL;
    }

    PyArrayArrayConverterObject *self = PyObject_NewVar(
            PyArrayArrayConverterObject, cls, narrs);
    if (self == NULL) {
        return NULL;
    }
    PyObject_InitVar((PyVarObject *)self, &PyArrayArrayConverter_Type, narrs);

    self->narrs = 0;
    self->flags = 0;
    self->wrap = NULL;
    self->wrap_type = NULL;

    if (narrs == 0) {
        return (PyObject *)self;
    }
    self->flags = (NPY_CH_ALL_PYSCALARS | NPY_CH_ALL_SCALARS);

    creation_item *item = self->items;
    /* increase self->narrs in loop for cleanup */
    for (int i = 0; i < narrs; i++, item++) {
        item->object = PyTuple_GET_ITEM(args, i);

        /* Fast path if input is an array (maybe FromAny should be faster): */
        if (PyArray_Check(item->object)) {
            Py_INCREF(item->object);
            item->array = (PyArrayObject *)item->object;
            item->scalar_input = 0;
        }
        else {
            item->array = (PyArrayObject *)PyArray_FromAny_int(
                    item->object, NULL, NULL, 0, 0, 0, NULL,
                    &item->scalar_input);
            if (item->array == NULL) {
                goto fail;
            }
        }

        /* At this point, assume cleanup should happen for this item */
        self->narrs++;
        Py_INCREF(item->object);
        item->DType = NPY_DTYPE(PyArray_DESCR(item->array));
        Py_INCREF(item->DType);

        /*
         * Check whether we were passed a an int/float/complex Python scalar.
         * If not, set `descr` and clear pyscalar/scalar flags as needed.
         */
        if (item->scalar_input && npy_mark_tmp_array_if_pyscalar(
                item->object, item->array, &item->DType)) {
            item->descr = NULL;
            /* Do not mark the stored array: */
            ((PyArrayObject_fields *)(item->array))->flags &= (
                    ~NPY_ARRAY_WAS_PYTHON_LITERAL);
        }
        else {
            item->descr = PyArray_DESCR(item->array);
            Py_INCREF(item->descr);

            if (item->scalar_input) {
                self->flags &= ~NPY_CH_ALL_PYSCALARS;
            }
            else {
                self->flags &= ~(NPY_CH_ALL_PYSCALARS | NPY_CH_ALL_SCALARS);
            }
        }
    }

    return (PyObject *)self;

  fail:
    Py_DECREF(self);
    return NULL;
}


static PyObject *
array_converter_get_scalar_input(PyArrayArrayConverterObject *self)
{
    PyObject *ret = PyTuple_New(self->narrs);
    if (ret == NULL) {
        return NULL;
    }

    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        if (item->scalar_input) {
            Py_INCREF(Py_True);
            PyTuple_SET_ITEM(ret, i, Py_True);
        }
        else {
            Py_INCREF(Py_False);
            PyTuple_SET_ITEM(ret, i, Py_False);
        }
    }
    return ret;
}


static int
find_wrap(PyArrayArrayConverterObject *self)
{
    if (self->wrap != NULL) {
        return 0;  /* nothing to do */
    }

    /* Allocate scratch space (could be optimized away) */
    PyObject **objects = PyMem_Malloc(self->narrs * sizeof(PyObject *));
    if (objects == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < self->narrs; i++) {
        objects[i] = self->items[i].object;
    }
    int ret = npy_find_array_wrap(
            self->narrs, objects, &self->wrap, &self->wrap_type);
    PyMem_FREE(objects);
    return ret;
}


typedef enum {
    CONVERT = 0,
    PRESERVE = 1,
    CONVERT_IF_NO_ARRAY = 2,
} scalar_policy;


static int
pyscalar_mode_conv(PyObject *obj, scalar_policy *policy)
{
    PyObject *strings[3] = {
            npy_interned_str.convert, npy_interned_str.preserve,
            npy_interned_str.convert_if_no_array};

    /* First quick pass using the identity (should practically always match) */
    for (int i = 0; i < 3; i++) {
        if (obj == strings[i]) {
            *policy = i;
            return 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        int cmp = PyObject_RichCompareBool(obj, strings[i], Py_EQ);
        if (cmp < 0) {
            return 0;
        }
        if (cmp) {
            *policy = i;
            return 1;
        }
    }
    PyErr_SetString(PyExc_ValueError,
            "invalid pyscalar mode, must be 'convert', 'preserve', or "
            "'convert_if_no_array' (default).");
    return 0;
}


static PyObject *
array_converter_as_arrays(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    npy_bool subok = NPY_TRUE;
    scalar_policy policy = CONVERT_IF_NO_ARRAY;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("as_arrays", args, len_args, kwnames,
            "$subok", &PyArray_BoolConverter, &subok,
            /* how to handle scalars (ignored if dtype is given). */
            "$pyscalars", &pyscalar_mode_conv, &policy,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (policy == CONVERT_IF_NO_ARRAY) {
        if (self->flags & NPY_CH_ALL_PYSCALARS) {
            policy = CONVERT;
        }
        else {
            policy = PRESERVE;
        }
    }

    PyObject *res = PyTuple_New(self->narrs);
    if (res == NULL) {
        return NULL;
    }
    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        PyObject *res_item;
        if (item->descr == NULL && policy == PRESERVE) {
            res_item = item->object;
            Py_INCREF(res_item);
        }
        else {
            res_item = (PyObject *)item->array;
            Py_INCREF(res_item);
            if (!subok) {
                /* PyArray_EnsureArray steals the reference... */
                res_item = PyArray_EnsureArray(res_item);
                if (res_item == NULL) {
                    goto fail;
                }
            }
        }

        if (PyTuple_SetItem(res, i, res_item) < 0) {
            goto fail;
        }
    }

    return res;

  fail:
    Py_DECREF(res);
    return NULL;
}


static PyObject *
array_converter_wrap(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    PyObject *to_scalar = Py_None;
    npy_bool ensure_scalar;

    if (find_wrap(self) < 0) {
        return NULL;
    }

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("wrap", args, len_args, kwnames,
            "", NULL, &obj,
            /* Three-way "bool", if `None` inspect input to decide. */
            "$to_scalar", NULL, &to_scalar,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (to_scalar == Py_None) {
        ensure_scalar = self->flags & NPY_CH_ALL_SCALARS;
    }
    else {
        if (!PyArray_BoolConverter(to_scalar, &ensure_scalar)) {
            return NULL;
        }
    }

    return npy_apply_wrap(
        obj, NULL, self->wrap, self->wrap_type, NULL, ensure_scalar, NPY_FALSE);
}


static PyObject *
array_converter_result_type(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArray_Descr *result = NULL;
    npy_dtype_info dt_info = {NULL, NULL};
    npy_bool ensure_inexact = NPY_FALSE;

    /* Allocate scratch space (could be optimized away) */
    void *DTypes_and_descrs = PyMem_Malloc(
            ((self->narrs + 1) * 2) * sizeof(PyObject *));
    if (DTypes_and_descrs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    PyArray_DTypeMeta **DTypes = DTypes_and_descrs;
    PyArray_Descr **descrs = (PyArray_Descr **)(DTypes + self->narrs + 1);

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("result_type", args, len_args, kwnames,
            "|extra_dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|ensure_inexact", &PyArray_BoolConverter, &ensure_inexact,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }

    int ndescrs = 0;
    int nDTypes = 0;
    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        DTypes[nDTypes] = item->DType;
        nDTypes++;
        if (item->descr != NULL) {
            descrs[ndescrs] = item->descr;
            ndescrs++;
        }
    }

    if (ensure_inexact) {
        if (dt_info.dtype != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "extra_dtype and ensure_inexact are mutually exclusive.");
            goto finish;
        }
        Py_INCREF(&PyArray_PyFloatDType);
        dt_info.dtype = &PyArray_PyFloatDType;
    }

    if (dt_info.dtype != NULL) {
        DTypes[nDTypes] = dt_info.dtype;
        nDTypes++;
    }
    if (dt_info.descr != NULL) {
        descrs[ndescrs] = dt_info.descr;
        ndescrs++;
    }

    PyArray_DTypeMeta *common_dtype = PyArray_PromoteDTypeSequence(
            nDTypes, DTypes);
    if (common_dtype == NULL) {
        goto finish;
    }
    if (ndescrs == 0) {
        result = NPY_DT_CALL_default_descr(common_dtype);
    }
    else {
        result = PyArray_CastToDTypeAndPromoteDescriptors(
                ndescrs, descrs, common_dtype);
    }
    Py_DECREF(common_dtype);

  finish:
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    PyMem_Free(DTypes_and_descrs);
    return (PyObject *)result;
}


static PyGetSetDef array_converter_getsets[] = {
    {"scalar_input",
        (getter)array_converter_get_scalar_input,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};


static PyMethodDef array_converter_methods[] = {
    {"as_arrays", 
        (PyCFunction)array_converter_as_arrays,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"result_type",
        (PyCFunction)array_converter_result_type,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"wrap",
        (PyCFunction)array_converter_wrap,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};


static void
array_converter_dealloc(PyArrayArrayConverterObject *self)
{
    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        Py_XDECREF(item->array);
        Py_XDECREF(item->object);
        Py_XDECREF(item->DType);
        Py_XDECREF(item->descr);
    }

    Py_XDECREF(self->wrap);
    Py_XDECREF(self->wrap_type);
    PyObject_Del((PyObject *)self);
}


static Py_ssize_t
array_converter_length(PyArrayArrayConverterObject *self)
{
    return self->narrs;
}


static PyObject *
array_converter_item(PyArrayArrayConverterObject *self, Py_ssize_t item)
{
    /* Python ensures no negative indices (and probably the below also) */
    if (item < 0 || item >= self->narrs) {
        PyErr_SetString(PyExc_IndexError, "invalid index");
        return NULL;
    }

    /* Follow the `as_arrays` default of `CONVERT_IF_NO_ARRAY`: */
    PyObject *res;
    if (self->items[item].descr == NULL
            && !(self->flags & NPY_CH_ALL_PYSCALARS)) {
        res = self->items[item].object;
    }
    else {
        res = (PyObject *)self->items[item].array;
    }

    Py_INCREF(res);
    return res;
}


static PySequenceMethods array_converter_as_sequence = {
    .sq_length = (lenfunc)array_converter_length,
    .sq_item = (ssizeargfunc)array_converter_item,
};


NPY_NO_EXPORT PyTypeObject PyArrayArrayConverter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._core._multiarray_umath._array_converter",
    .tp_basicsize = sizeof(PyArrayArrayConverterObject),
    .tp_itemsize = sizeof(creation_item),
    .tp_new = array_converter_new,
    .tp_dealloc = (destructor)array_converter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = array_converter_getsets,
    .tp_methods = array_converter_methods,
    .tp_as_sequence = &array_converter_as_sequence,
};

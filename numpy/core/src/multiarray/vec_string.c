#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <ctype.h>

#include "numpy/arrayobject.h"
#include "dtypemeta.h"

static PyObject *
_vec_string_is_alpha(PyArrayIterObject *in_iter, npy_intp itemsize)
{
    char *data = in_iter->dataptr;
    for (int i = 0; i < itemsize; i++) {
        if (!isalpha(data[i])) {
            Py_RETURN_FALSE;
        }
    }
    Py_RETURN_TRUE;
}

typedef PyObject * (*_vec_string_fast_op)(PyArrayIterObject *, npy_intp);

typedef struct {
    const char *name;
    _vec_string_fast_op func;
    int with_args;
} _vec_string_named_fast_op;

static _vec_string_named_fast_op *SUPPORTED_FAST_OPS[] = {
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {{NULL, NULL, -1}},
    (_vec_string_named_fast_op[]) {
        {"isalpha", _vec_string_is_alpha, 0},
    },
};

static int N_FAST_OP_LISTS = 8;

int
get_fast_op(PyObject *method_name, _vec_string_fast_op *method, int *with_args)
{
    const char *method_name_str = PyUnicode_AsUTF8(method_name);
    const int method_name_len = strlen(method_name_str);

    if (method_name_len >= N_FAST_OP_LISTS) {
        return 0;
    }

    for (_vec_string_named_fast_op *f = SUPPORTED_FAST_OPS[method_name_len]; f->name != NULL; f++) {
        if (strncmp(method_name_str, f->name, method_name_len) == 0) {
            *method = f->func;
            *with_args = f->with_args;
            return 1;
        }
    }

    return 0;
}

/*
 * returns 1 if array is a user-defined string dtype, sets an error and
 * returns 0 otherwise
 */
static int _is_user_defined_string_array(PyArrayObject* array)
{
    if (NPY_DT_is_user_defined(PyArray_DESCR(array))) {
        PyTypeObject* scalar_type = NPY_DTYPE(PyArray_DESCR(array))->scalar_type;
        if (PyType_IsSubtype(scalar_type, &PyBytes_Type) ||
            PyType_IsSubtype(scalar_type, &PyUnicode_Type)) {
            return 1;
        }
        else {
            PyErr_SetString(
                PyExc_TypeError,
                "string comparisons are only allowed for dtypes with a "
                "scalar type that is a subtype of str or bytes.");
            return 0;
        }
    }
    else {
        PyErr_SetString(
            PyExc_TypeError,
            "string operation on non-string array");
        return 0;
    }
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    Py_ssize_t i, n, nargs;

    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        PyObject* item = PySequence_GetItem(args, i-1);
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        broadcast_args[i] = item;
        Py_DECREF(item);
    }
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    n = in_iter->numiter;

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->nd,
            in_iter->dimensions, type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* args_tuple = PyTuple_New(n);
        if (args_tuple == NULL) {
            goto err;
        }

        for (i = 0; i < n; i++) {
            PyArrayIterObject* it = in_iter->iters[i];
            PyObject* arg = PyArray_ToScalar(PyArray_ITER_DATA(it), it->ao);
            if (arg == NULL) {
                Py_DECREF(args_tuple);
                goto err;
            }
            /* Steals ref to arg */
            PyTuple_SetItem(args_tuple, i, arg);
        }

        item_result = PyObject_CallObject(method, args_tuple);
        Py_DECREF(args_tuple);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;

    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, in_iter->ao);
        if (item == NULL) {
            goto err;
        }

        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_fast_op_no_args(PyArrayObject *char_array, PyArray_Descr *type, _vec_string_fast_op method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    npy_intp itemsize;

    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    itemsize = PyArray_ITEMSIZE(char_array);
    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result = method(in_iter, itemsize);

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

NPY_NO_EXPORT PyObject *
_vec_string(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyArrayObject* char_array = NULL;
    PyArray_Descr *type;
    PyObject* method_name;
    PyObject* args_seq = NULL;

    int fast_method_found = 0;
    int fast_method_with_args = 0;
    _vec_string_fast_op fast_method = NULL;
    PyObject* method = NULL;
    PyObject* result = NULL;

    if (!PyArg_ParseTuple(args, "O&O&O|O",
                PyArray_Converter, &char_array,
                PyArray_DescrConverter, &type,
                &method_name, &args_seq)) {
        goto err;
    }

    fast_method_found = get_fast_op(method_name, &fast_method, &fast_method_with_args);
    if (fast_method_found && PyArray_TYPE(char_array) != NPY_UNICODE) {
        // if (fast_method_with_args) {
        //     result = _vec_string_fast_op_with_args(char_array, type, fast_method, args_seq);
        // } else {
            result = _vec_string_fast_op_no_args(char_array, type, fast_method);
        // }
    } else {
        if (PyArray_TYPE(char_array) == NPY_STRING) {
            method = PyObject_GetAttr((PyObject *)&PyBytes_Type, method_name);
        }
        else if (PyArray_TYPE(char_array) == NPY_UNICODE) {
            method = PyObject_GetAttr((PyObject *)&PyUnicode_Type, method_name);
        }
        else {
            if (_is_user_defined_string_array(char_array)) {
                PyTypeObject* scalar_type =
                    NPY_DTYPE(PyArray_DESCR(char_array))->scalar_type;
                method = PyObject_GetAttr((PyObject*)scalar_type, method_name);
            }
            else {
                Py_DECREF(type);
                goto err;
            }
        }
        if (method == NULL) {
            Py_DECREF(type);
            goto err;
        }

        if (args_seq == NULL
                || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
            result = _vec_string_no_args(char_array, type, method);
        }
        else if (PySequence_Check(args_seq)) {
            result = _vec_string_with_args(char_array, type, method, args_seq);
        }
        else {
            Py_DECREF(type);
            PyErr_SetString(PyExc_TypeError,
                    "'args' must be a sequence of arguments");
            goto err;
        }
    }
    if (result == NULL) {
        goto err;
    }

    Py_DECREF(char_array);
    Py_XDECREF(method);

    return (PyObject*)result;

 err:
    Py_XDECREF(char_array);
    Py_XDECREF(method);

    return 0;
}

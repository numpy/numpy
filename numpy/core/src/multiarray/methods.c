#define PY_SSIZE_T_CLEAN
#include <stdarg.h>
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"
#include "ctors.h"
#include "calculation.h"
#include "convert_datatype.h"
#include "item_selection.h"
#include "conversion_utils.h"
#include "shape.h"

#include "methods.h"


/* NpyArg_ParseKeywords
 *
 * Utility function that provides the keyword parsing functionality of
 * PyArg_ParseTupleAndKeywords without having to have an args argument.
 *
 */
static int
NpyArg_ParseKeywords(PyObject *keys, const char *format, char **kwlist, ...)
{
    PyObject *args = PyTuple_New(0);
    int ret;
    va_list va;

    if (args == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed to allocate new tuple");
        return 0;
    }
    va_start(va, kwlist);
    ret = PyArg_VaParseTupleAndKeywords(args, keys, format, kwlist, va);
    va_end(va);
    Py_DECREF(args);
    return ret;
}

static PyObject *
get_forwarding_ndarray_method(const char *name)
{
    PyObject *module_methods, *callable;

    /* Get a reference to the function we're calling */
    module_methods = PyImport_ImportModule("numpy.core._methods");
    if (module_methods == NULL) {
        return NULL;
    }
    callable = PyDict_GetItemString(PyModule_GetDict(module_methods), name);
    if (callable == NULL) {
        Py_DECREF(module_methods);
        PyErr_Format(PyExc_RuntimeError,
                "NumPy internal error: could not find function "
                "numpy.core._methods.%s", name);
    }

    Py_INCREF(callable);
    Py_DECREF(module_methods);
    return callable;
}

/*
 * Forwards an ndarray method to a the Python function
 * numpy.core._methods.<name>(...)
 */
static PyObject *
forward_ndarray_method(PyArrayObject *self, PyObject *args, PyObject *kwds,
                            PyObject *forwarding_callable)
{
    PyObject *sargs, *ret;
    int i, n;

    /* Combine 'self' and 'args' together into one tuple */
    n = PyTuple_GET_SIZE(args);
    sargs = PyTuple_New(n + 1);
    if (sargs == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    PyTuple_SET_ITEM(sargs, 0, (PyObject *)self);
    for (i = 0; i < n; ++i) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(sargs, i+1, item);
    }

    /* Call the function and return */
    ret = PyObject_Call(forwarding_callable, sargs, kwds);
    Py_DECREF(sargs);
    return ret;
}

/*
 * Forwards an ndarray method to the function numpy.core._methods.<name>(...),
 * caching the callable in a local static variable. Note that the
 * initialization is not thread-safe, but relies on the CPython GIL to
 * be correct.
 */
#define NPY_FORWARD_NDARRAY_METHOD(name) \
        static PyObject *callable = NULL; \
        if (callable == NULL) { \
            callable = get_forwarding_ndarray_method(name); \
            if (callable == NULL) { \
                return NULL; \
            } \
        } \
        return forward_ndarray_method(self, args, kwds, callable)


static PyObject *
array_take(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int dimension = NPY_MAXDIMS;
    PyObject *indices;
    PyArrayObject *out = NULL;
    NPY_CLIPMODE mode = NPY_RAISE;
    static char *kwlist[] = {"indices", "axis", "out", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O&", kwlist,
                                     &indices,
                                     PyArray_AxisConverter, &dimension,
                                     PyArray_OutputConverter, &out,
                                     PyArray_ClipmodeConverter, &mode))
        return NULL;

    return PyArray_Return((PyArrayObject *)
                PyArray_TakeFrom(self, indices, dimension, out, mode));
}

static PyObject *
array_fill(PyArrayObject *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (PyArray_FillWithScalar(self, obj) < 0) {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_put(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices, *values;
    NPY_CLIPMODE mode = NPY_RAISE;
    static char *kwlist[] = {"indices", "values", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&", kwlist,
                                     &indices,
                                     &values,
                                     PyArray_ClipmodeConverter, &mode))
        return NULL;
    return PyArray_PutTo(self, values, indices, mode);
}

static PyObject *
array_reshape(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"order", NULL};
    PyArray_Dims newshape;
    PyObject *ret;
    NPY_ORDER order = NPY_CORDER;
    Py_ssize_t n = PyTuple_Size(args);

    if (!NpyArg_ParseKeywords(kwds, "|O&", keywords,
                PyArray_OrderConverter, &order)) {
        return NULL;
    }

    if (n <= 1) {
        if (PyTuple_GET_ITEM(args, 0) == Py_None) {
            return PyArray_View(self, NULL, NULL);
        }
        if (!PyArg_ParseTuple(args, "O&", PyArray_IntpConverter,
                              &newshape)) {
            return NULL;
        }
    }
    else {
        if (!PyArray_IntpConverter(args, &newshape)) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                                "invalid shape");
            }
            goto fail;
        }
    }
    ret = PyArray_Newshape(self, &newshape, order);
    PyDimMem_FREE(newshape.ptr);
    return ret;

 fail:
    PyDimMem_FREE(newshape.ptr);
    return NULL;
}

static PyObject *
array_squeeze(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *axis_in = NULL;
    npy_bool axis_flags[NPY_MAXDIMS];

    static char *kwlist[] = {"axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &axis_in)) {
        return NULL;
    }

    if (axis_in == NULL || axis_in == Py_None) {
        return PyArray_Squeeze(self);
    }
    else {
        if (PyArray_ConvertMultiAxis(axis_in, PyArray_NDIM(self),
                                            axis_flags) != NPY_SUCCEED) {
            return NULL;
        }

        return PyArray_SqueezeSelected(self, axis_flags);
    }
}

static PyObject *
array_view(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *out_dtype = NULL;
    PyObject *out_type = NULL;
    PyArray_Descr *dtype = NULL;

    static char *kwlist[] = {"dtype", "type", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
                                     &out_dtype,
                                     &out_type)) {
        return NULL;
    }

    /* If user specified a positional argument, guess whether it
       represents a type or a dtype for backward compatibility. */
    if (out_dtype) {
        /* type specified? */
        if (PyType_Check(out_dtype) &&
            PyType_IsSubtype((PyTypeObject *)out_dtype,
                             &PyArray_Type)) {
            if (out_type) {
                PyErr_SetString(PyExc_ValueError,
                                "Cannot specify output type twice.");
                return NULL;
            }
            out_type = out_dtype;
            out_dtype = NULL;
        }
    }

    if ((out_type) && (!PyType_Check(out_type) ||
                       !PyType_IsSubtype((PyTypeObject *)out_type,
                                         &PyArray_Type))) {
        PyErr_SetString(PyExc_ValueError,
                        "Type must be a sub-type of ndarray type");
        return NULL;
    }

    if ((out_dtype) &&
        (PyArray_DescrConverter(out_dtype, &dtype) == NPY_FAIL)) {
        return NULL;
    }

    return PyArray_View(self, dtype, (PyTypeObject*)out_type);
}

static PyObject *
array_argmax(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out))
        return NULL;

    return PyArray_Return((PyArrayObject *)PyArray_ArgMax(self, axis, out));
}

static PyObject *
array_argmin(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out))
        return NULL;

    return PyArray_Return((PyArrayObject *)PyArray_ArgMin(self, axis, out));
}

static PyObject *
array_max(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_amax");
}

static PyObject *
array_min(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_amin");
}

static PyObject *
array_ptp(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out))
        return NULL;

    return PyArray_Ptp(self, axis, out);
}


static PyObject *
array_swapaxes(PyArrayObject *self, PyObject *args)
{
    int axis1, axis2;

    if (!PyArg_ParseTuple(args, "ii", &axis1, &axis2)) {
        return NULL;
    }
    return PyArray_SwapAxes(self, axis1, axis2);
}


/* steals typed reference */
/*NUMPY_API
  Get a subset of bytes from each element of the array
*/
NPY_NO_EXPORT PyObject *
PyArray_GetField(PyArrayObject *self, PyArray_Descr *typed, int offset)
{
    PyObject *ret = NULL;

    if (offset < 0 || (offset + typed->elsize) > PyArray_DESCR(self)->elsize) {
        PyErr_Format(PyExc_ValueError,
                     "Need 0 <= offset <= %d for requested type "
                     "but received offset = %d",
                     PyArray_DESCR(self)->elsize-typed->elsize, offset);
        Py_DECREF(typed);
        return NULL;
    }
    ret = PyArray_NewFromDescr(Py_TYPE(self),
                               typed,
                               PyArray_NDIM(self), PyArray_DIMS(self),
                               PyArray_STRIDES(self),
                               PyArray_BYTES(self) + offset,
                               PyArray_FLAGS(self)&(~NPY_ARRAY_F_CONTIGUOUS),
                               (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(((PyArrayObject *)ret), (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    PyArray_UpdateFlags((PyArrayObject *)ret, NPY_ARRAY_UPDATE_ALL);
    return ret;
}

static PyObject *
array_getfield(PyArrayObject *self, PyObject *args, PyObject *kwds)
{

    PyArray_Descr *dtype = NULL;
    int offset = 0;
    static char *kwlist[] = {"dtype", "offset", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
                                     PyArray_DescrConverter, &dtype,
                                     &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    return PyArray_GetField(self, dtype, offset);
}


/*NUMPY_API
  Set a subset of bytes from each element of the array
*/
NPY_NO_EXPORT int
PyArray_SetField(PyArrayObject *self, PyArray_Descr *dtype,
                 int offset, PyObject *val)
{
    PyObject *ret = NULL;
    int retval = 0;

    if (offset < 0 || (offset + dtype->elsize) > PyArray_DESCR(self)->elsize) {
        PyErr_Format(PyExc_ValueError,
                     "Need 0 <= offset <= %d for requested type "
                     "but received offset = %d",
                     PyArray_DESCR(self)->elsize-dtype->elsize, offset);
        Py_DECREF(dtype);
        return -1;
    }
    ret = PyArray_NewFromDescr(Py_TYPE(self),
                           dtype, PyArray_NDIM(self), PyArray_DIMS(self),
                           PyArray_STRIDES(self), PyArray_BYTES(self) + offset,
                           PyArray_FLAGS(self), (PyObject *)self);
    if (ret == NULL) {
        return -1;
    }

    PyArray_UpdateFlags((PyArrayObject *)ret, NPY_ARRAY_UPDATE_ALL);
    retval = PyArray_CopyObject((PyArrayObject *)ret, val);
    Py_DECREF(ret);
    return retval;
}

static PyObject *
array_setfield(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *dtype = NULL;
    int offset = 0;
    PyObject *value;
    static char *kwlist[] = {"value", "dtype", "offset", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|i", kwlist,
                                     &value,
                                     PyArray_DescrConverter, &dtype,
                                     &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        PyErr_SetString(PyExc_RuntimeError,
                    "cannot call setfield on an object array");
        Py_DECREF(dtype);
        return NULL;
    }

    if (PyArray_SetField(self, dtype, offset, value) < 0) {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

/* This doesn't change the descriptor just the actual data...
 */

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_Byteswap(PyArrayObject *self, npy_bool inplace)
{
    PyArrayObject *ret;
    npy_intp size;
    PyArray_CopySwapNFunc *copyswapn;
    PyArrayIterObject *it;

    copyswapn = PyArray_DESCR(self)->f->copyswapn;
    if (inplace) {
        if (PyArray_FailUnlessWriteable(self, "array to be byte-swapped") < 0) {
            return NULL;
        }
        size = PyArray_SIZE(self);
        if (PyArray_ISONESEGMENT(self)) {
            copyswapn(PyArray_DATA(self), PyArray_DESCR(self)->elsize, NULL, -1, size, 1, self);
        }
        else { /* Use iterator */
            int axis = -1;
            npy_intp stride;
            it = (PyArrayIterObject *)                      \
                PyArray_IterAllButAxis((PyObject *)self, &axis);
            stride = PyArray_STRIDES(self)[axis];
            size = PyArray_DIMS(self)[axis];
            while (it->index < it->size) {
                copyswapn(it->dataptr, stride, NULL, -1, size, 1, self);
                PyArray_ITER_NEXT(it);
            }
            Py_DECREF(it);
        }

        Py_INCREF(self);
        return (PyObject *)self;
    }
    else {
        PyObject *new;
        if ((ret = (PyArrayObject *)PyArray_NewCopy(self,-1)) == NULL) {
            return NULL;
        }
        new = PyArray_Byteswap(ret, NPY_TRUE);
        Py_DECREF(new);
        return (PyObject *)ret;
    }
}


static PyObject *
array_byteswap(PyArrayObject *self, PyObject *args)
{
    npy_bool inplace = NPY_FALSE;

    if (!PyArg_ParseTuple(args, "|O&",
                            PyArray_BoolConverter, &inplace)) {
        return NULL;
    }
    return PyArray_Byteswap(self, inplace);
}

static PyObject *
array_tolist(PyArrayObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyArray_ToList(self);
}


static PyObject *
array_tostring(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                                     PyArray_OrderConverter, &order)) {
        return NULL;
    }
    return PyArray_ToString(self, order);
}


/* This should grow an order= keyword to be consistent
 */

static PyObject *
array_tofile(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int own;
    PyObject *file;
    FILE *fd;
    char *sep = "";
    char *format = "";
    static char *kwlist[] = {"file", "sep", "format", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ss", kwlist,
                                     &file,
                                     &sep,
                                     &format)) {
        return NULL;
    }

    if (PyBytes_Check(file) || PyUnicode_Check(file)) {
        file = npy_PyFile_OpenFile(file, "wb");
        if (file == NULL) {
            return NULL;
        }
        own = 1;
    }
    else {
        Py_INCREF(file);
        own = 0;
    }

    fd = npy_PyFile_Dup(file, "wb");
    if (fd == NULL) {
        PyErr_SetString(PyExc_IOError,
                "first argument must be a string or open file");
        goto fail;
    }
    if (PyArray_ToFile(self, fd, sep, format) < 0) {
        goto fail;
    }
    if (npy_PyFile_DupClose(file, fd) < 0) {
        goto fail;
    }
    if (own && npy_PyFile_CloseFile(file) < 0) {
        goto fail;
    }
    Py_DECREF(file);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_DECREF(file);
    return NULL;
}

static PyObject *
array_toscalar(PyArrayObject *self, PyObject *args)
{
    npy_intp multi_index[NPY_MAXDIMS];
    int n = PyTuple_GET_SIZE(args);
    int idim, ndim = PyArray_NDIM(self);

    /* If there is a tuple as a single argument, treat it as the argument */
    if (n == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        n = PyTuple_GET_SIZE(args);
    }

    if (n == 0) {
        if (PyArray_SIZE(self) == 1) {
            for (idim = 0; idim < ndim; ++idim) {
                multi_index[idim] = 0;
            }
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "can only convert an array of size 1 to a Python scalar");
            return NULL;
        }
    }
    /* Special case of C-order flat indexing... :| */
    else if (n == 1 && ndim != 1) {
        npy_intp *shape = PyArray_SHAPE(self);
        npy_intp value, size = PyArray_SIZE(self);

        value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }

        if (check_and_adjust_index(&value, size, -1) < 0) {
            return NULL;
        }

        /* Convert the flat index into a multi-index */
        for (idim = ndim-1; idim >= 0; --idim) {
            multi_index[idim] = value % shape[idim];
            value /= shape[idim];
        }
    }
    /* A multi-index tuple */
    else if (n == ndim) {
        npy_intp value;

        for (idim = 0; idim < ndim; ++idim) {
            value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, idim));
            if (value == -1 && PyErr_Occurred()) {
                return NULL;
            }
            multi_index[idim] = value;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "incorrect number of indices for array");
        return NULL;
    }

    return PyArray_MultiIndexGetItem(self, multi_index);
}

static PyObject *
array_setscalar(PyArrayObject *self, PyObject *args)
{
    npy_intp multi_index[NPY_MAXDIMS];
    int n = PyTuple_GET_SIZE(args) - 1;
    int idim, ndim = PyArray_NDIM(self);
    PyObject *obj;

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError,
                "itemset must have at least one argument");
        return NULL;
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return NULL;
    }

    obj = PyTuple_GET_ITEM(args, n);

    /* If there is a tuple as a single argument, treat it as the argument */
    if (n == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        n = PyTuple_GET_SIZE(args);
    }

    if (n == 0) {
        if (PyArray_SIZE(self) == 1) {
            for (idim = 0; idim < ndim; ++idim) {
                multi_index[idim] = 0;
            }
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "can only convert an array of size 1 to a Python scalar");
        }
    }
    /* Special case of C-order flat indexing... :| */
    else if (n == 1 && ndim != 1) {
        npy_intp *shape = PyArray_SHAPE(self);
        npy_intp value, size = PyArray_SIZE(self);

        value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }

        if (check_and_adjust_index(&value, size, -1) < 0) {
            return NULL;
        }

        /* Convert the flat index into a multi-index */
        for (idim = ndim-1; idim >= 0; --idim) {
            multi_index[idim] = value % shape[idim];
            value /= shape[idim];
        }
    }
    /* A multi-index tuple */
    else if (n == ndim) {
        npy_intp value;

        for (idim = 0; idim < ndim; ++idim) {
            value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, idim));
            if (value == -1 && PyErr_Occurred()) {
                return NULL;
            }
            multi_index[idim] = value;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "incorrect number of indices for array");
        return NULL;
    }

    if (PyArray_MultiIndexSetItem(self, multi_index, obj) < 0) {
        return NULL;
    }
    else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}

static PyObject *
array_astype(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dtype", "order", "casting",
                             "subok", "copy", NULL};
    PyArray_Descr *dtype = NULL;
    /*
     * TODO: UNSAFE default for compatibility, I think
     *       switching to SAME_KIND by default would be good.
     */
    NPY_CASTING casting = NPY_UNSAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    int forcecopy = 1, subok = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&ii", kwlist,
                            PyArray_DescrConverter, &dtype,
                            PyArray_OrderConverter, &order,
                            PyArray_CastingConverter, &casting,
                            &subok,
                            &forcecopy)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    /*
     * If the memory layout matches and, data types are equivalent,
     * and it's not a subtype if subok is False, then we
     * can skip the copy.
     */
    if (!forcecopy && (order == NPY_KEEPORDER ||
                       (order == NPY_ANYORDER &&
                            (PyArray_IS_C_CONTIGUOUS(self) ||
                            PyArray_IS_F_CONTIGUOUS(self))) ||
                       (order == NPY_CORDER &&
                            PyArray_IS_C_CONTIGUOUS(self)) ||
                       (order == NPY_FORTRANORDER &&
                            PyArray_IS_F_CONTIGUOUS(self))) &&
                    (subok || PyArray_CheckExact(self)) &&
                    PyArray_EquivTypes(dtype, PyArray_DESCR(self))) {
        Py_DECREF(dtype);
        Py_INCREF(self);
        return (PyObject *)self;
    }
    else if (PyArray_CanCastArrayTo(self, dtype, casting)) {
        PyArrayObject *ret;

        /* If the requested dtype is flexible, adapt it */
        PyArray_AdaptFlexibleDType((PyObject *)self, PyArray_DESCR(self),
                                                                    &dtype);
        if (dtype == NULL) {
            return NULL;
        }

        /* This steals the reference to dtype, so no DECREF of dtype */
        ret = (PyArrayObject *)PyArray_NewLikeArray(
                                    self, order, dtype, subok);
        if (ret == NULL) {
            return NULL;
        }

        if (PyArray_CopyInto(ret, self) < 0) {
            Py_DECREF(ret);
            return NULL;
        }

        return (PyObject *)ret;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast array from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyArray_DESCR(self)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)dtype));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        Py_DECREF(dtype);
        return NULL;
    }
}

/* default sub-type implementation */


static PyObject *
array_wraparray(PyArrayObject *self, PyObject *args)
{
    PyArrayObject *arr, *ret;
    PyObject *obj;

    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "only accepts 1 argument");
        return NULL;
    }
    obj = PyTuple_GET_ITEM(args, 0);
    if (obj == NULL) {
        return NULL;
    }
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "can only be called with ndarray object");
        return NULL;
    }
    arr = (PyArrayObject *)obj;

    if (Py_TYPE(self) != Py_TYPE(arr)){
        PyArray_Descr *dtype = PyArray_DESCR(arr);
        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                   dtype,
                                   PyArray_NDIM(arr),
                                   PyArray_DIMS(arr),
                                   PyArray_STRIDES(arr), PyArray_DATA(arr),
                                   PyArray_FLAGS(arr), (PyObject *)self);
        if (ret == NULL) {
            return NULL;
        }
        Py_INCREF(obj);
        if (PyArray_SetBaseObject(ret, obj) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        return (PyObject *)ret;
    } else {
        /*The type was set in __array_prepare__*/
        Py_INCREF(arr);
        return (PyObject *)arr;
    }
}


static PyObject *
array_preparearray(PyArrayObject *self, PyObject *args)
{
    PyObject *obj;
    PyArrayObject *arr, *ret;
    PyArray_Descr *dtype;

    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "only accepts 1 argument");
        return NULL;
    }
    obj = PyTuple_GET_ITEM(args, 0);
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "can only be called with ndarray object");
        return NULL;
    }
    arr = (PyArrayObject *)obj;

    if (Py_TYPE(self) == Py_TYPE(arr)) {
        /* No need to create a new view */
        Py_INCREF(arr);
        return (PyObject *)arr;
    }

    dtype = PyArray_DESCR(arr);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                               dtype,
                               PyArray_NDIM(arr),
                               PyArray_DIMS(arr),
                               PyArray_STRIDES(arr), PyArray_DATA(arr),
                               PyArray_FLAGS(arr), (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(arr);
    if (PyArray_SetBaseObject(ret, (PyObject *)arr) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}


static PyObject *
array_getarray(PyArrayObject *self, PyObject *args)
{
    PyArray_Descr *newtype = NULL;
    PyObject *ret;

    if (!PyArg_ParseTuple(args, "|O&",
                            PyArray_DescrConverter, &newtype)) {
        Py_XDECREF(newtype);
        return NULL;
    }

    /* convert to PyArray_Type */
    if (!PyArray_CheckExact(self)) {
        PyArrayObject *new;
        PyTypeObject *subtype = &PyArray_Type;

        if (!PyType_IsSubtype(Py_TYPE(self), &PyArray_Type)) {
            subtype = &PyArray_Type;
        }

        Py_INCREF(PyArray_DESCR(self));
        new = (PyArrayObject *)PyArray_NewFromDescr(subtype,
                                   PyArray_DESCR(self),
                                   PyArray_NDIM(self),
                                   PyArray_DIMS(self),
                                   PyArray_STRIDES(self),
                                   PyArray_DATA(self),
                                   PyArray_FLAGS(self), NULL);
        if (new == NULL) {
            return NULL;
        }
        Py_INCREF(self);
        PyArray_SetBaseObject(new, (PyObject *)self);
        self = new;
    }
    else {
        Py_INCREF(self);
    }

    if ((newtype == NULL) || PyArray_EquivTypes(PyArray_DESCR(self), newtype)) {
        return (PyObject *)self;
    }
    else {
        ret = PyArray_CastToType(self, newtype, 0);
        Py_DECREF(self);
        return ret;
    }
}


static PyObject *
array_copy(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                                     PyArray_OrderConverter, &order)) {
        return NULL;
    }

    return PyArray_NewCopy(self, order);
}

/* Separate from array_copy to make __copy__ preserve Fortran contiguity. */
static PyObject *
array_copy_keeporder(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyArray_NewCopy(self, NPY_KEEPORDER);
}

#include <stdio.h>
static PyObject *
array_resize(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"refcheck", NULL};
    Py_ssize_t size = PyTuple_Size(args);
    int refcheck = 1;
    PyArray_Dims newshape;
    PyObject *ret, *obj;


    if (!NpyArg_ParseKeywords(kwds, "|i", kwlist,  &refcheck)) {
        return NULL;
    }

    if (size == 0) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else if (size == 1) {
        obj = PyTuple_GET_ITEM(args, 0);
        if (obj == Py_None) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        args = obj;
    }
    if (!PyArray_IntpConverter(args, &newshape)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "invalid shape");
        }
        return NULL;
    }

    ret = PyArray_Resize(self, &newshape, refcheck, NPY_CORDER);
    PyDimMem_FREE(newshape.ptr);
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_repeat(PyArrayObject *self, PyObject *args, PyObject *kwds) {
    PyObject *repeats;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"repeats", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
                                     &repeats,
                                     PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_Repeat(self, repeats, axis));
}

static PyObject *
array_choose(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"out", "mode", NULL};
    PyObject *choices;
    PyArrayObject *out = NULL;
    NPY_CLIPMODE clipmode = NPY_RAISE;
    Py_ssize_t n = PyTuple_Size(args);

    if (n <= 1) {
        if (!PyArg_ParseTuple(args, "O", &choices)) {
            return NULL;
        }
    }
    else {
        choices = args;
    }

    if (!NpyArg_ParseKeywords(kwds, "|O&O&", keywords,
                PyArray_OutputConverter, &out,
                PyArray_ClipmodeConverter, &clipmode)) {
        return NULL;
    }

    return PyArray_Return((PyArrayObject *)PyArray_Choose(self, choices, out, clipmode));
}

static PyObject *
array_sort(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis=-1;
    int val;
    NPY_SORTKIND sortkind = NPY_QUICKSORT;
    PyObject *order = NULL;
    PyArray_Descr *saved = NULL;
    PyArray_Descr *newd;
    static char *kwlist[] = {"axis", "kind", "order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&O", kwlist,
                                    &axis,
                                    PyArray_SortkindConverter, &sortkind,
                                    &order)) {
        return NULL;
    }
    if (order == Py_None) {
        order = NULL;
    }
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                            "order when the array has no fields.");
            return NULL;
        }
        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        newd = PyArray_DescrNew(saved);
        Py_DECREF(newd->names);
        newd->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    val = PyArray_Sort(self, axis, sortkind);
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    if (val < 0) {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_partition(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis=-1;
    int val;
    NPY_SELECTKIND sortkind = NPY_INTROSELECT;
    PyObject *order = NULL;
    PyArray_Descr *saved = NULL;
    PyArray_Descr *newd;
    static char *kwlist[] = {"kth", "axis", "kind", "order", NULL};
    PyArrayObject * ktharray;
    PyObject * kthobj;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO&O", kwlist,
                                     &kthobj,
                                     &axis,
                                     PyArray_SelectkindConverter, &sortkind,
                                     &order)) {
        return NULL;
    }

    if (order == Py_None) {
        order = NULL;
    }
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                            "order when the array has no fields.");
            return NULL;
        }
        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        newd = PyArray_DescrNew(saved);
        Py_DECREF(newd->names);
        newd->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    ktharray = (PyArrayObject *)PyArray_FromAny(kthobj, NULL, 0, 1,
                                                NPY_ARRAY_DEFAULT, NULL);
    if (ktharray == NULL)
        return NULL;

    val = PyArray_Partition(self, ktharray, axis, sortkind);
    Py_DECREF(ktharray);

    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    if (val < 0) {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_argsort(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = -1;
    NPY_SORTKIND sortkind = NPY_QUICKSORT;
    PyObject *order = NULL, *res;
    PyArray_Descr *newd, *saved=NULL;
    static char *kwlist[] = {"axis", "kind", "order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_SortkindConverter, &sortkind,
                                     &order)) {
        return NULL;
    }
    if (order == Py_None) {
        order = NULL;
    }
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify "
                            "order when the array has no fields.");
            return NULL;
        }
        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        newd = PyArray_DescrNew(saved);
        newd->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    res = PyArray_ArgSort(self, axis, sortkind);
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    return PyArray_Return((PyArrayObject *)res);
}


static PyObject *
array_argpartition(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = -1;
    NPY_SELECTKIND sortkind = NPY_INTROSELECT;
    PyObject *order = NULL, *res;
    PyArray_Descr *newd, *saved=NULL;
    static char *kwlist[] = {"kth", "axis", "kind", "order", NULL};
    PyObject * kthobj;
    PyArrayObject * ktharray;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O", kwlist,
                                     &kthobj,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_SelectkindConverter, &sortkind,
                                     &order)) {
        return NULL;
    }
    if (order == Py_None) {
        order = NULL;
    }
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify "
                            "order when the array has no fields.");
            return NULL;
        }
        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        newd = PyArray_DescrNew(saved);
        newd->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    ktharray = (PyArrayObject *)PyArray_FromAny(kthobj, NULL, 0, 1,
                                                NPY_ARRAY_DEFAULT, NULL);
    if (ktharray == NULL)
        return NULL;

    res = PyArray_ArgPartition(self, ktharray, axis, sortkind);
    Py_DECREF(ktharray);

    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    return PyArray_Return((PyArrayObject *)res);
}

static PyObject *
array_searchsorted(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"keys", "side", "sorter", NULL};
    PyObject *keys;
    PyObject *sorter;
    NPY_SEARCHSIDE side = NPY_SEARCHLEFT;

    sorter = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O:searchsorted",
                                     kwlist, &keys,
                                     PyArray_SearchsideConverter, &side, &sorter)) {
        return NULL;
    }
    if (sorter == Py_None) {
        sorter = NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_SearchSorted(self, keys, side, sorter));
}

static void
_deepcopy_call(char *iptr, char *optr, PyArray_Descr *dtype,
               PyObject *deepcopy, PyObject *visit)
{
    if (!PyDataType_REFCHK(dtype)) {
        return;
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            _deepcopy_call(iptr + offset, optr + offset, new,
                           deepcopy, visit);
        }
    }
    else {
        PyObject *itemp, *otemp;
        PyObject *res;
        NPY_COPY_PYOBJECT_PTR(&itemp, iptr);
        NPY_COPY_PYOBJECT_PTR(&otemp, optr);
        Py_XINCREF(itemp);
        /* call deepcopy on this argument */
        res = PyObject_CallFunctionObjArgs(deepcopy, itemp, visit, NULL);
        Py_XDECREF(itemp);
        Py_XDECREF(otemp);
        NPY_COPY_PYOBJECT_PTR(optr, &res);
    }

}


static PyObject *
array_deepcopy(PyArrayObject *self, PyObject *args)
{
    PyObject* visit;
    char *optr;
    PyArrayIterObject *it;
    PyObject *copy, *deepcopy;
    PyArrayObject *ret;

    if (!PyArg_ParseTuple(args, "O", &visit)) {
        return NULL;
    }
    ret = (PyArrayObject *)PyArray_NewCopy(self, NPY_KEEPORDER);
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        copy = PyImport_ImportModule("copy");
        if (copy == NULL) {
            return NULL;
        }
        deepcopy = PyObject_GetAttrString(copy, "deepcopy");
        Py_DECREF(copy);
        if (deepcopy == NULL) {
            return NULL;
        }
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
        if (it == NULL) {
            Py_DECREF(deepcopy);
            return NULL;
        }
        optr = PyArray_DATA(ret);
        while(it->index < it->size) {
            _deepcopy_call(it->dataptr, optr, PyArray_DESCR(self), deepcopy, visit);
            optr += PyArray_DESCR(self)->elsize;
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(deepcopy);
        Py_DECREF(it);
    }
    return PyArray_Return(ret);
}

/* Convert Array to flat list (using getitem) */
static PyObject *
_getlist_pkl(PyArrayObject *self)
{
    PyObject *theobject;
    PyArrayIterObject *iter = NULL;
    PyObject *list;
    PyArray_GetItemFunc *getitem;

    getitem = PyArray_DESCR(self)->f->getitem;
    iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (iter == NULL) {
        return NULL;
    }
    list = PyList_New(iter->size);
    if (list == NULL) {
        Py_DECREF(iter);
        return NULL;
    }
    while (iter->index < iter->size) {
        theobject = getitem(iter->dataptr, self);
        PyList_SET_ITEM(list, (int) iter->index, theobject);
        PyArray_ITER_NEXT(iter);
    }
    Py_DECREF(iter);
    return list;
}

static int
_setlist_pkl(PyArrayObject *self, PyObject *list)
{
    PyObject *theobject;
    PyArrayIterObject *iter = NULL;
    PyArray_SetItemFunc *setitem;

    setitem = PyArray_DESCR(self)->f->setitem;
    iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (iter == NULL) {
        return -1;
    }
    while(iter->index < iter->size) {
        theobject = PyList_GET_ITEM(list, (int) iter->index);
        setitem(theobject, iter->dataptr, self);
        PyArray_ITER_NEXT(iter);
    }
    Py_XDECREF(iter);
    return 0;
}


static PyObject *
array_reduce(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    /* version number of this pickle type. Increment if we need to
       change the format. Be sure to handle the old versions in
       array_setstate. */
    const int version = 1;
    PyObject *ret = NULL, *state = NULL, *obj = NULL, *mod = NULL;
    PyObject *mybool, *thestr = NULL;
    PyArray_Descr *descr;

    /* Return a tuple of (callable object, arguments, object's state) */
    /*  We will put everything in the object's state, so that on UnPickle
        it can use the string object as memory without a copy */

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy.core.multiarray");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    obj = PyObject_GetAttrString(mod, "_reconstruct");
    Py_DECREF(mod);
    PyTuple_SET_ITEM(ret, 0, obj);
    PyTuple_SET_ITEM(ret, 1,
                     Py_BuildValue("ONc",
                                   (PyObject *)Py_TYPE(self),
                                   Py_BuildValue("(N)",
                                                 PyInt_FromLong(0)),
                                   /* dummy data-type */
                                   'b'));

    /* Now fill in object's state.  This is a tuple with
       5 arguments

       1) an integer with the pickle version.
       2) a Tuple giving the shape
       3) a PyArray_Descr Object (with correct bytorder set)
       4) a npy_bool stating if Fortran or not
       5) a Python object representing the data (a string, or
       a list or any user-defined object).

       Notice because Python does not describe a mechanism to write
       raw data to the pickle, this performs a copy to a string first
    */

    state = PyTuple_New(5);
    if (state == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));
    PyTuple_SET_ITEM(state, 1, PyObject_GetAttrString((PyObject *)self,
                                                      "shape"));
    descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    PyTuple_SET_ITEM(state, 2, (PyObject *)descr);
    mybool = (PyArray_ISFORTRAN(self) ? Py_True : Py_False);
    Py_INCREF(mybool);
    PyTuple_SET_ITEM(state, 3, mybool);
    if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_LIST_PICKLE)) {
        thestr = _getlist_pkl(self);
    }
    else {
        thestr = PyArray_ToString(self, NPY_ANYORDER);
    }
    if (thestr == NULL) {
        Py_DECREF(ret);
        Py_DECREF(state);
        return NULL;
    }
    PyTuple_SET_ITEM(state, 4, thestr);
    PyTuple_SET_ITEM(ret, 2, state);
    return ret;
}

static PyObject *
array_setstate(PyArrayObject *self, PyObject *args)
{
    PyObject *shape;
    PyArray_Descr *typecode;
    int version = 1;
    int is_f_order;
    PyObject *rawdata = NULL;
    char *datastr;
    Py_ssize_t len;
    npy_intp size, dimensions[NPY_MAXDIMS];
    int nd;

    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    /* This will free any memory associated with a and
       use the string in setstate as the (writeable) memory.
    */
    if (!PyArg_ParseTuple(args, "(iO!O!iO)",
                            &version,
                            &PyTuple_Type, &shape,
                            &PyArrayDescr_Type, &typecode,
                            &is_f_order,
                            &rawdata)) {
        PyErr_Clear();
        version = 0;
        if (!PyArg_ParseTuple(args, "(O!O!iO)",
                            &PyTuple_Type, &shape,
                            &PyArrayDescr_Type, &typecode,
                            &is_f_order,
                            &rawdata)) {
            return NULL;
        }
    }

    /* If we ever need another pickle format, increment the version
       number. But we should still be able to handle the old versions.
       We've only got one right now. */
    if (version != 1 && version != 0) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.ndarray pickle",
                     version);
        return NULL;
    }

    Py_XDECREF(PyArray_DESCR(self));
    fa->descr = typecode;
    Py_INCREF(typecode);
    nd = PyArray_IntpFromSequence(shape, dimensions, NPY_MAXDIMS);
    if (nd < 0) {
        return NULL;
    }
    size = PyArray_MultiplyList(dimensions, nd);
    if (PyArray_DESCR(self)->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid data-type size.");
        return NULL;
    }
    if (size < 0 || size > NPY_MAX_INTP / PyArray_DESCR(self)->elsize) {
        PyErr_NoMemory();
        return NULL;
    }

    if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        if (!PyList_Check(rawdata)) {
            PyErr_SetString(PyExc_TypeError,
                            "object pickle not returning list");
            return NULL;
        }
    }
    else {
        Py_INCREF(rawdata);

#if defined(NPY_PY3K)
        /* Backward compatibility with Python 2 Numpy pickles */
        if (PyUnicode_Check(rawdata)) {
            PyObject *tmp;
            tmp = PyUnicode_AsLatin1String(rawdata);
            Py_DECREF(rawdata);
            rawdata = tmp;
        }
#endif

        if (!PyBytes_Check(rawdata)) {
            PyErr_SetString(PyExc_TypeError,
                            "pickle not returning string");
            Py_DECREF(rawdata);
            return NULL;
        }

        if (PyBytes_AsStringAndSize(rawdata, &datastr, &len)) {
            Py_DECREF(rawdata);
            return NULL;
        }

        if ((len != (PyArray_DESCR(self)->elsize * size))) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size does not"  \
                            " match array size");
            Py_DECREF(rawdata);
            return NULL;
        }
    }

    if ((PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA)) {
        if (PyArray_DATA(self) != NULL) {
            PyDataMem_FREE(PyArray_DATA(self));
        }
        PyArray_CLEARFLAGS(self, NPY_ARRAY_OWNDATA);
    }
    Py_XDECREF(PyArray_BASE(self));
    fa->base = NULL;

    PyArray_CLEARFLAGS(self, NPY_ARRAY_UPDATEIFCOPY);

    if (PyArray_DIMS(self) != NULL) {
        PyDimMem_FREE(PyArray_DIMS(self));
        fa->dimensions = NULL;
    }

    fa->flags = NPY_ARRAY_DEFAULT;

    fa->nd = nd;

    if (nd > 0) {
        fa->dimensions = PyDimMem_NEW(3*nd);
        fa->strides = PyArray_DIMS(self) + nd;
        memcpy(PyArray_DIMS(self), dimensions, sizeof(npy_intp)*nd);
        _array_fill_strides(PyArray_STRIDES(self), dimensions, nd,
                               PyArray_DESCR(self)->elsize,
                               (is_f_order ? NPY_ARRAY_F_CONTIGUOUS :
                                             NPY_ARRAY_C_CONTIGUOUS),
                               &(fa->flags));
    }

    if (!PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        int swap=!PyArray_ISNOTSWAPPED(self);
        fa->data = datastr;
#ifndef NPY_PY3K
        /* Check that the string is not interned */
        if (!_IsAligned(self) || swap || PyString_CHECK_INTERNED(rawdata)) {
#else
        /* Bytes should always be considered immutable, but we just grab the
         * pointer if they are large, to save memory. */
        if (!_IsAligned(self) || swap || (len <= 1000)) {
#endif
            npy_intp num = PyArray_NBYTES(self);
            fa->data = PyDataMem_NEW(num);
            if (PyArray_DATA(self) == NULL) {
                fa->nd = 0;
                PyDimMem_FREE(PyArray_DIMS(self));
                Py_DECREF(rawdata);
                return PyErr_NoMemory();
            }
            if (swap) {
                /* byte-swap on pickle-read */
                npy_intp numels = num / PyArray_DESCR(self)->elsize;
                PyArray_DESCR(self)->f->copyswapn(PyArray_DATA(self),
                                        PyArray_DESCR(self)->elsize,
                                        datastr, PyArray_DESCR(self)->elsize,
                                        numels, 1, self);
                if (!PyArray_ISEXTENDED(self)) {
                    fa->descr = PyArray_DescrFromType(
                                    PyArray_DESCR(self)->type_num);
                }
                else {
                    fa->descr = PyArray_DescrNew(typecode);
                    if (PyArray_DESCR(self)->byteorder == NPY_BIG) {
                        PyArray_DESCR(self)->byteorder = NPY_LITTLE;
                    }
                    else if (PyArray_DESCR(self)->byteorder == NPY_LITTLE) {
                        PyArray_DESCR(self)->byteorder = NPY_BIG;
                    }
                }
                Py_DECREF(typecode);
            }
            else {
                memcpy(PyArray_DATA(self), datastr, num);
            }
            PyArray_ENABLEFLAGS(self, NPY_ARRAY_OWNDATA);
            fa->base = NULL;
            Py_DECREF(rawdata);
        }
        else {
            if (PyArray_SetBaseObject(self, rawdata) < 0) {
                return NULL;
            }
        }
    }
    else {
        fa->data = PyDataMem_NEW(PyArray_NBYTES(self));
        if (PyArray_DATA(self) == NULL) {
            fa->nd = 0;
            fa->data = PyDataMem_NEW(PyArray_DESCR(self)->elsize);
            if (PyArray_DIMS(self)) {
                PyDimMem_FREE(PyArray_DIMS(self));
            }
            return PyErr_NoMemory();
        }
        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_NEEDS_INIT)) {
            memset(PyArray_DATA(self), 0, PyArray_NBYTES(self));
        }
        PyArray_ENABLEFLAGS(self, NPY_ARRAY_OWNDATA);
        fa->base = NULL;
        if (_setlist_pkl(self, rawdata) < 0) {
            return NULL;
        }
    }

    PyArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);

    Py_INCREF(Py_None);
    return Py_None;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Dump(PyObject *self, PyObject *file, int protocol)
{
    PyObject *cpick = NULL;
    PyObject *ret;
    if (protocol < 0) {
        protocol = 2;
    }

#if defined(NPY_PY3K)
    cpick = PyImport_ImportModule("pickle");
#else
    cpick = PyImport_ImportModule("cPickle");
#endif
    if (cpick == NULL) {
        return -1;
    }
    if (PyBytes_Check(file) || PyUnicode_Check(file)) {
        file = npy_PyFile_OpenFile(file, "wb");
        if (file == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(file);
    }
    ret = PyObject_CallMethod(cpick, "dump", "OOi", self, file, protocol);
    Py_XDECREF(ret);
    Py_DECREF(file);
    Py_DECREF(cpick);
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_Dumps(PyObject *self, int protocol)
{
    PyObject *cpick = NULL;
    PyObject *ret;
    if (protocol < 0) {
        protocol = 2;
    }
#if defined(NPY_PY3K)
    cpick = PyImport_ImportModule("pickle");
#else
    cpick = PyImport_ImportModule("cPickle");
#endif
    if (cpick == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(cpick, "dumps", "Oi", self, protocol);
    Py_DECREF(cpick);
    return ret;
}


static PyObject *
array_dump(PyArrayObject *self, PyObject *args)
{
    PyObject *file = NULL;
    int ret;

    if (!PyArg_ParseTuple(args, "O", &file)) {
        return NULL;
    }
    ret = PyArray_Dump((PyObject *)self, file, 2);
    if (ret < 0) {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
array_dumps(PyArrayObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyArray_Dumps((PyObject *)self, 2);
}


static PyObject *
array_transpose(PyArrayObject *self, PyObject *args)
{
    PyObject *shape = Py_None;
    Py_ssize_t n = PyTuple_Size(args);
    PyArray_Dims permute;
    PyObject *ret;

    if (n > 1) {
        shape = args;
    }
    else if (n == 1) {
        shape = PyTuple_GET_ITEM(args, 0);
    }

    if (shape == Py_None) {
        ret = PyArray_Transpose(self, NULL);
    }
    else {
        if (!PyArray_IntpConverter(shape, &permute)) {
            return NULL;
        }
        ret = PyArray_Transpose(self, &permute);
        PyDimMem_FREE(permute.ptr);
    }

    return ret;
}

#define _CHKTYPENUM(typ) ((typ) ? (typ)->type_num : NPY_NOTYPE)

static PyObject *
array_mean(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_mean");
}

static PyObject *
array_sum(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_sum");
}


static PyObject *
array_cumsum(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArray_Descr *dtype = NULL;
    PyArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"axis", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyArray_CumSum(self, axis, rtype, out);
}

static PyObject *
array_prod(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_prod");
}

static PyObject *
array_cumprod(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArray_Descr *dtype = NULL;
    PyArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"axis", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyArray_CumProd(self, axis, rtype, out);
}


static PyObject *
array_dot(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *fname, *ret, *b, *out = NULL;
    static PyObject *numpycore = NULL;
    char * kwords[] = {"b", "out", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwords, &b, &out)) { 
        return NULL;
    }

    /* Since blas-dot is exposed only on the Python side, we need to grab it
     * from there */
    if (numpycore == NULL) {
        numpycore = PyImport_ImportModule("numpy.core");
        if (numpycore == NULL) {
            return NULL;
        }
    }
    fname = PyUString_FromString("dot");
    if (out == NULL) {
        ret = PyObject_CallMethodObjArgs(numpycore, fname, self, b, NULL);
    }
    else {
        ret = PyObject_CallMethodObjArgs(numpycore, fname, self, b, out, NULL);
    }
    Py_DECREF(fname);
    return ret;
}


static PyObject *
array_any(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_any");
}


static PyObject *
array_all(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_all");
}

static PyObject *
array_stddev(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_std");
}

static PyObject *
array_variance(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_FORWARD_NDARRAY_METHOD("_var");
}

static PyObject *
array_compress(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyObject *condition;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"condition", "axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&", kwlist,
                                     &condition,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }
    return PyArray_Return(
                (PyArrayObject *)PyArray_Compress(self, condition, axis, out));
}


static PyObject *
array_nonzero(PyArrayObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyArray_Nonzero(self);
}


static PyObject *
array_trace(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis1 = 0, axis2 = 1, offset = 0;
    PyArray_Descr *dtype = NULL;
    PyArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"offset", "axis1", "axis2", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiO&O&", kwlist,
                                     &offset,
                                     &axis1,
                                     &axis2,
                                     PyArray_DescrConverter2, &dtype,
                                     PyArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyArray_Return((PyArrayObject *)PyArray_Trace(self, offset, axis1, axis2, rtype, out));
}

#undef _CHKTYPENUM


static PyObject *
array_clip(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *min = NULL, *max = NULL;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"min", "max", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO&", kwlist,
                                     &min,
                                     &max,
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }
    if (max == NULL && min == NULL) {
        PyErr_SetString(PyExc_ValueError, "One of max or min must be given.");
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_Clip(self, min, max, out));
}


static PyObject *
array_conjugate(PyArrayObject *self, PyObject *args)
{

    PyArrayObject *out = NULL;
    if (!PyArg_ParseTuple(args, "|O&",
                          PyArray_OutputConverter,
                          &out)) {
        return NULL;
    }
    return PyArray_Conjugate(self, out);
}


static PyObject *
array_diagonal(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis1 = 0, axis2 = 1, offset = 0;
    static char *kwlist[] = {"offset", "axis1", "axis2", NULL};
    PyArrayObject *ret;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &offset,
                                     &axis1,
                                     &axis2)) {
        return NULL;
    }

    ret = (PyArrayObject *)PyArray_Diagonal(self, offset, axis1, axis2);
    return PyArray_Return(ret);
}


static PyObject *
array_flatten(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                            PyArray_OrderConverter, &order)) {
        return NULL;
    }
    return PyArray_Flatten(self, order);
}


static PyObject *
array_ravel(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                            PyArray_OrderConverter, &order)) {
        return NULL;
    }
    return PyArray_Ravel(self, order);
}


static PyObject *
array_round(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int decimals = 0;
    PyArrayObject *out = NULL;
    static char *kwlist[] = {"decimals", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&", kwlist,
                                     &decimals,
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_Round(self, decimals, out));
}



static PyObject *
array_setflags(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"write", "align", "uic", NULL};
    PyObject *write_flag = Py_None;
    PyObject *align_flag = Py_None;
    PyObject *uic = Py_None;
    int flagback = PyArray_FLAGS(self);

    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
                                     &write_flag,
                                     &align_flag,
                                     &uic))
        return NULL;

    if (align_flag != Py_None) {
        if (PyObject_Not(align_flag)) {
            PyArray_CLEARFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        else if (_IsAligned(self)) {
            PyArray_ENABLEFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                            "cannot set aligned flag of mis-"\
                            "aligned array to True");
            return NULL;
        }
    }

    if (uic != Py_None) {
        if (PyObject_IsTrue(uic)) {
            fa->flags = flagback;
            PyErr_SetString(PyExc_ValueError,
                            "cannot set UPDATEIFCOPY "       \
                            "flag to True");
            return NULL;
        }
        else {
            PyArray_CLEARFLAGS(self, NPY_ARRAY_UPDATEIFCOPY);
            Py_XDECREF(fa->base);
            fa->base = NULL;
        }
    }

    if (write_flag != Py_None) {
        if (PyObject_IsTrue(write_flag)) {
            if (_IsWriteable(self)) {
                PyArray_ENABLEFLAGS(self, NPY_ARRAY_WRITEABLE);
            }
            else {
                fa->flags = flagback;
                PyErr_SetString(PyExc_ValueError,
                                "cannot set WRITEABLE "
                                "flag to True of this "
                                "array");
                return NULL;
            }
        }
        else {
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEABLE);
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
array_newbyteorder(PyArrayObject *self, PyObject *args)
{
    char endian = NPY_SWAP;
    PyArray_Descr *new;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_ByteorderConverter,
                          &endian)) {
        return NULL;
    }
    new = PyArray_DescrNewByteorder(PyArray_DESCR(self), endian);
    if (!new) {
        return NULL;
    }
    return PyArray_View(self, new, NULL);

}

NPY_NO_EXPORT PyMethodDef array_methods[] = {

    /* for subtypes */
    {"__array__",
        (PyCFunction)array_getarray,
        METH_VARARGS, NULL},
    {"__array_prepare__",
        (PyCFunction)array_preparearray,
        METH_VARARGS, NULL},
    {"__array_wrap__",
        (PyCFunction)array_wraparray,
        METH_VARARGS, NULL},

    /* for the copy module */
    {"__copy__",
        (PyCFunction)array_copy_keeporder,
        METH_VARARGS, NULL},
    {"__deepcopy__",
        (PyCFunction)array_deepcopy,
        METH_VARARGS, NULL},

    /* for Pickling */
    {"__reduce__",
        (PyCFunction) array_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction) array_setstate,
        METH_VARARGS, NULL},
    {"dumps",
        (PyCFunction) array_dumps,
        METH_VARARGS, NULL},
    {"dump",
        (PyCFunction) array_dump,
        METH_VARARGS, NULL},

    /* Original and Extended methods added 2005 */
    {"all",
        (PyCFunction)array_all,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"any",
        (PyCFunction)array_any,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmax",
        (PyCFunction)array_argmax,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmin",
        (PyCFunction)array_argmin,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argpartition",
        (PyCFunction)array_argpartition,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argsort",
        (PyCFunction)array_argsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"astype",
        (PyCFunction)array_astype,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"byteswap",
        (PyCFunction)array_byteswap,
        METH_VARARGS, NULL},
    {"choose",
        (PyCFunction)array_choose,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip",
        (PyCFunction)array_clip,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"compress",
        (PyCFunction)array_compress,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"conj",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    {"conjugate",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    {"copy",
        (PyCFunction)array_copy,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumprod",
        (PyCFunction)array_cumprod,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumsum",
        (PyCFunction)array_cumsum,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"diagonal",
        (PyCFunction)array_diagonal,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"dot",
        (PyCFunction)array_dot,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fill",
        (PyCFunction)array_fill,
        METH_VARARGS, NULL},
    {"flatten",
        (PyCFunction)array_flatten,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"getfield",
        (PyCFunction)array_getfield,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"item",
        (PyCFunction)array_toscalar,
        METH_VARARGS, NULL},
    {"itemset",
        (PyCFunction) array_setscalar,
        METH_VARARGS, NULL},
    {"max",
        (PyCFunction)array_max,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean",
        (PyCFunction)array_mean,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"min",
        (PyCFunction)array_min,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"newbyteorder",
        (PyCFunction)array_newbyteorder,
        METH_VARARGS, NULL},
    {"nonzero",
        (PyCFunction)array_nonzero,
        METH_VARARGS, NULL},
    {"partition",
        (PyCFunction)array_partition,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"prod",
        (PyCFunction)array_prod,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ptp",
        (PyCFunction)array_ptp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"put",
        (PyCFunction)array_put,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel",
        (PyCFunction)array_ravel,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"repeat",
        (PyCFunction)array_repeat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape",
        (PyCFunction)array_reshape,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"resize",
        (PyCFunction)array_resize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"round",
        (PyCFunction)array_round,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"searchsorted",
        (PyCFunction)array_searchsorted,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"setfield",
        (PyCFunction)array_setfield,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"setflags",
        (PyCFunction)array_setflags,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"sort",
        (PyCFunction)array_sort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"squeeze",
        (PyCFunction)array_squeeze,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"std",
        (PyCFunction)array_stddev,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"sum",
        (PyCFunction)array_sum,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"swapaxes",
        (PyCFunction)array_swapaxes,
        METH_VARARGS, NULL},
    {"take",
        (PyCFunction)array_take,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"tofile",
        (PyCFunction)array_tofile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"tolist",
        (PyCFunction)array_tolist,
        METH_VARARGS, NULL},
    {"tostring",
        (PyCFunction)array_tostring,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"trace",
        (PyCFunction)array_trace,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"transpose",
        (PyCFunction)array_transpose,
        METH_VARARGS, NULL},
    {"var",
        (PyCFunction)array_variance,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"view",
        (PyCFunction)array_view,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

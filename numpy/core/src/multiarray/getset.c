/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"

#include "common.h"
#include "conversion_utils.h"
#include "ctors.h"
#include "scalartypes.h"
#include "descriptor.h"
#include "getset.h"
#include "arrayobject.h"
#include "mem_overlap.h"
#include "alloc.h"
#include "npy_buffer.h"

/*******************  array attribute get and set routines ******************/

static PyObject *
array_ndim_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(PyArray_NDIM(self));
}

static PyObject *
array_flags_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_NewFlagsObject((PyObject *)self);
}

static PyObject *
array_shape_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_DIMS(self));
}


static int
array_shape_set(PyArrayObject *self, PyObject *val, void* NPY_UNUSED(ignored))
{
    int nd;
    PyArrayObject *ret;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array shape");
        return -1;
    }
    /* Assumes C-order */
    ret = (PyArrayObject *)PyArray_Reshape(self, val);
    if (ret == NULL) {
        return -1;
    }
    if (PyArray_DATA(ret) != PyArray_DATA(self)) {
        Py_DECREF(ret);
        PyErr_SetString(PyExc_AttributeError,
                        "Incompatible shape for in-place modification. Use "
                        "`.reshape()` to make a copy with the desired shape.");
        return -1;
    }

    nd = PyArray_NDIM(ret);
    if (nd > 0) {
        /* create new dimensions and strides */
        npy_intp *_dimensions = npy_alloc_cache_dim(2 * nd);
        if (_dimensions == NULL) {
            Py_DECREF(ret);
            PyErr_NoMemory();
            return -1;
        }
        /* Free old dimensions and strides */
        npy_free_cache_dim_array(self);
        ((PyArrayObject_fields *)self)->nd = nd;
        ((PyArrayObject_fields *)self)->dimensions = _dimensions; 
        ((PyArrayObject_fields *)self)->strides = _dimensions + nd;

        if (nd) {
            memcpy(PyArray_DIMS(self), PyArray_DIMS(ret), nd*sizeof(npy_intp));
            memcpy(PyArray_STRIDES(self), PyArray_STRIDES(ret), nd*sizeof(npy_intp));
        }
    }
    else {
        /* Free old dimensions and strides */
        npy_free_cache_dim_array(self);        
        ((PyArrayObject_fields *)self)->nd = 0;
        ((PyArrayObject_fields *)self)->dimensions = NULL;
        ((PyArrayObject_fields *)self)->strides = NULL;
    }

    Py_DECREF(ret);
    PyArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return 0;
}


static PyObject *
array_strides_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_STRIDES(self));
}

static int
array_strides_set(PyArrayObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    PyArray_Dims newstrides = {NULL, -1};
    PyArrayObject *new;
    npy_intp numbytes = 0;
    npy_intp offset = 0;
    npy_intp lower_offset = 0;
    npy_intp upper_offset = 0;
    Py_buffer view;

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array strides");
        return -1;
    }
    if (!PyArray_OptionalIntpConverter(obj, &newstrides) ||
        newstrides.len == -1) {
        PyErr_SetString(PyExc_TypeError, "invalid strides");
        return -1;
    }
    if (newstrides.len != PyArray_NDIM(self)) {
        PyErr_Format(PyExc_ValueError, "strides must be "       \
                     " same length as shape (%d)", PyArray_NDIM(self));
        goto fail;
    }
    new = self;
    while(PyArray_BASE(new) && PyArray_Check(PyArray_BASE(new))) {
        new = (PyArrayObject *)(PyArray_BASE(new));
    }
    /*
     * Get the available memory through the buffer interface on
     * PyArray_BASE(new) or if that fails from the current new
     */
    if (PyArray_BASE(new) &&
            PyObject_GetBuffer(PyArray_BASE(new), &view, PyBUF_SIMPLE) >= 0) {
        offset = PyArray_BYTES(self) - (char *)view.buf;
        numbytes = view.len + offset;
        PyBuffer_Release(&view);
    }
    else {
        PyErr_Clear();
        offset_bounds_from_strides(PyArray_ITEMSIZE(new), PyArray_NDIM(new),
                                   PyArray_DIMS(new), PyArray_STRIDES(new),
                                   &lower_offset, &upper_offset);

        offset = PyArray_BYTES(self) - (PyArray_BYTES(new) + lower_offset);
        numbytes = upper_offset - lower_offset;
    }

    /* numbytes == 0 is special here, but the 0-size array case always works */
    if (!PyArray_CheckStrides(PyArray_ITEMSIZE(self), PyArray_NDIM(self),
                              numbytes, offset,
                              PyArray_DIMS(self), newstrides.ptr)) {
        PyErr_SetString(PyExc_ValueError, "strides is not "\
                        "compatible with available memory");
        goto fail;
    }
    if (newstrides.len) {
        memcpy(PyArray_STRIDES(self), newstrides.ptr, sizeof(npy_intp)*newstrides.len);
    }
    PyArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS |
                              NPY_ARRAY_ALIGNED);
    npy_free_cache_dim_obj(newstrides);
    return 0;

 fail:
    npy_free_cache_dim_obj(newstrides);
    return -1;
}



static PyObject *
array_priority_get(PyArrayObject *NPY_UNUSED(self), void *NPY_UNUSED(ignored))
{
    return PyFloat_FromDouble(NPY_PRIORITY);
}

static PyObject *
array_typestr_get(PyArrayObject *self)
{
    return arraydescr_protocol_typestr_get(PyArray_DESCR(self), NULL);
}

static PyObject *
array_descr_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    Py_INCREF(PyArray_DESCR(self));
    return (PyObject *)PyArray_DESCR(self);
}

static PyObject *
array_protocol_descr_get(PyArrayObject *self)
{
    PyObject *res;
    PyObject *dobj;

    res = arraydescr_protocol_descr_get(PyArray_DESCR(self), NULL);
    if (res) {
        return res;
    }
    PyErr_Clear();

    /* get default */
    dobj = PyTuple_New(2);
    if (dobj == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(dobj, 0, PyUnicode_FromString(""));
    PyTuple_SET_ITEM(dobj, 1, array_typestr_get(self));
    res = PyList_New(1);
    if (res == NULL) {
        Py_DECREF(dobj);
        return NULL;
    }
    PyList_SET_ITEM(res, 0, dobj);
    return res;
}

static PyObject *
array_protocol_strides_get(PyArrayObject *self)
{
    if (PyArray_ISCONTIGUOUS(self)) {
        Py_RETURN_NONE;
    }
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_STRIDES(self));
}



static PyObject *
array_dataptr_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return Py_BuildValue("NO",
                         PyLong_FromVoidPtr(PyArray_DATA(self)),
                         ((PyArray_FLAGS(self) & NPY_ARRAY_WRITEABLE) &&
                          !(PyArray_FLAGS(self) & NPY_ARRAY_WARN_ON_WRITE)) ?
                         Py_False : Py_True);
}

static PyObject *
array_ctypes_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *_numpy_internal;
    PyObject *ret;
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(_numpy_internal, "_ctypes", "ON", self,
                              PyLong_FromVoidPtr(PyArray_DATA(self)));
    Py_DECREF(_numpy_internal);
    return ret;
}

static PyObject *
array_interface_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *dict;
    PyObject *obj;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    int ret;

    /* dataptr */
    obj = array_dataptr_get(self, NULL);
    ret = PyDict_SetItemString(dict, "data", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    obj = array_protocol_strides_get(self);
    ret = PyDict_SetItemString(dict, "strides", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    obj = array_protocol_descr_get(self);
    ret = PyDict_SetItemString(dict, "descr", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    obj = arraydescr_protocol_typestr_get(PyArray_DESCR(self), NULL);
    ret = PyDict_SetItemString(dict, "typestr", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    obj = array_shape_get(self, NULL);
    ret = PyDict_SetItemString(dict, "shape", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    obj = PyLong_FromLong(3);
    ret = PyDict_SetItemString(dict, "version", obj);
    Py_DECREF(obj);
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    return dict;
}

static PyObject *
array_data_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyMemoryView_FromObject((PyObject *)self);
}

static int
array_data_set(PyArrayObject *self, PyObject *op, void *NPY_UNUSED(ignored))
{
    void *buf;
    Py_ssize_t buf_len;
    int writeable=1;
    Py_buffer view;

    /* 2016-19-02, 1.12 */
    int ret = DEPRECATE("Assigning the 'data' attribute is an "
                        "inherently unsafe operation and will "
                        "be removed in the future.");
    if (ret < 0) {
        return -1;
    }

    if (op == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array data");
        return -1;
    }
    if (PyObject_GetBuffer(op, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        writeable = 0;
        PyErr_Clear();
        if (PyObject_GetBuffer(op, &view, PyBUF_SIMPLE) < 0) {
            return -1;
        }
    }
    buf = view.buf;
    buf_len = view.len;
    /*
     * In Python 3 both of the deprecated functions PyObject_AsWriteBuffer and
     * PyObject_AsReadBuffer that this code replaces release the buffer. It is
     * up to the object that supplies the buffer to guarantee that the buffer
     * sticks around after the release.
     */
    PyBuffer_Release(&view);

    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_AttributeError,
                "cannot set single-segment buffer for discontiguous array");
        return -1;
    }
    if (PyArray_NBYTES(self) > buf_len) {
        PyErr_SetString(PyExc_AttributeError, "not enough data for array");
        return -1;
    }
    if (PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA) {
        PyArray_XDECREF(self);
        size_t nbytes = PyArray_NBYTES(self);
        if (nbytes == 0) {
            nbytes = 1;
        }
        PyObject *handler = PyArray_HANDLER(self);
        if (handler == NULL) {
            /* This can happen if someone arbitrarily sets NPY_ARRAY_OWNDATA */
            PyErr_SetString(PyExc_RuntimeError,
                            "no memory handler found but OWNDATA flag set");
            return -1;
        }
        PyDataMem_UserFREE(PyArray_DATA(self), nbytes, handler);
        Py_CLEAR(((PyArrayObject_fields *)self)->mem_handler);
    }
    if (PyArray_BASE(self)) {
        if (PyArray_FLAGS(self) & NPY_ARRAY_WRITEBACKIFCOPY) {
            PyArray_ENABLEFLAGS((PyArrayObject *)PyArray_BASE(self),
                                                NPY_ARRAY_WRITEABLE);
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEBACKIFCOPY);
        }
        Py_DECREF(PyArray_BASE(self));
        ((PyArrayObject_fields *)self)->base = NULL;
    }
    Py_INCREF(op);
    if (PyArray_SetBaseObject(self, op) < 0) {
        return -1;
    }
    ((PyArrayObject_fields *)self)->data = buf;
    ((PyArrayObject_fields *)self)->flags = NPY_ARRAY_CARRAY;
    if (!writeable) {
        PyArray_CLEARFLAGS(self, ~NPY_ARRAY_WRITEABLE);
    }
    return 0;
}


static PyObject *
array_itemsize_get(PyArrayObject *self, void* NPY_UNUSED(ignored))
{
    return PyLong_FromLong((long) PyArray_DESCR(self)->elsize);
}

static PyObject *
array_size_get(PyArrayObject *self, void* NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(PyArray_SIZE(self));
}

static PyObject *
array_nbytes_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(PyArray_NBYTES(self));
}


/*
 * If the type is changed.
 * Also needing change: strides, itemsize
 *
 * Either itemsize is exactly the same or the array is single-segment
 * (contiguous or fortran) with compatible dimensions The shape and strides
 * will be adjusted in that case as well.
 */
static int
array_descr_set(PyArrayObject *self, PyObject *arg, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *newtype = NULL;

    if (arg == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array dtype");
        return -1;
    }

    if (!(PyArray_DescrConverter(arg, &newtype)) ||
        newtype == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "invalid data-type for array");
        return -1;
    }

    /* check that we are not reinterpreting memory containing Objects. */
    if (_may_have_objects(PyArray_DESCR(self)) || _may_have_objects(newtype)) {
        static PyObject *checkfunc = NULL;
        PyObject *safe;

        npy_cache_import("numpy.core._internal", "_view_is_safe", &checkfunc);
        if (checkfunc == NULL) {
            goto fail;
        }

        safe = PyObject_CallFunction(checkfunc, "OO",
                                     PyArray_DESCR(self), newtype);
        if (safe == NULL) {
            goto fail;
        }
        Py_DECREF(safe);
    }

    /*
     * Viewing as an unsized void implies a void dtype matching the size of the
     * current dtype.
     */
    if (newtype->type_num == NPY_VOID &&
            PyDataType_ISUNSIZED(newtype) &&
            newtype->elsize != PyArray_DESCR(self)->elsize) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return -1;
        }
        newtype->elsize = PyArray_DESCR(self)->elsize;
    }

    /* Changing the size of the dtype results in a shape change */
    if (newtype->elsize != PyArray_DESCR(self)->elsize) {
        /* forbidden cases */
        if (PyArray_NDIM(self) == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Changing the dtype of a 0d array is only supported "
                    "if the itemsize is unchanged");
            goto fail;
        }
        else if (PyDataType_HASSUBARRAY(newtype)) {
            PyErr_SetString(PyExc_ValueError,
                    "Changing the dtype to a subarray type is only supported "
                    "if the total itemsize is unchanged");
            goto fail;
        }

        /* resize on last axis only */
        int axis = PyArray_NDIM(self) - 1;
        if (PyArray_DIMS(self)[axis] != 1 &&
                PyArray_SIZE(self) != 0 &&
                PyArray_STRIDES(self)[axis] != PyArray_DESCR(self)->elsize) {
            PyErr_SetString(PyExc_ValueError,
                    "To change to a dtype of a different size, the last axis "
                    "must be contiguous");
            goto fail;
        }

        npy_intp newdim;

        if (newtype->elsize < PyArray_DESCR(self)->elsize) {
            /* if it is compatible, increase the size of the last axis */
            if (newtype->elsize == 0 ||
                    PyArray_DESCR(self)->elsize % newtype->elsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "When changing to a smaller dtype, its size must be a "
                        "divisor of the size of original dtype");
                goto fail;
            }
            newdim = PyArray_DESCR(self)->elsize / newtype->elsize;
            PyArray_DIMS(self)[axis] *= newdim;
            PyArray_STRIDES(self)[axis] = newtype->elsize;
        }
        else /* newtype->elsize > PyArray_DESCR(self)->elsize */ {
            /* if it is compatible, decrease the size of the relevant axis */
            newdim = PyArray_DIMS(self)[axis] * PyArray_DESCR(self)->elsize;
            if ((newdim % newtype->elsize) != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "When changing to a larger dtype, its size must be a "
                        "divisor of the total size in bytes of the last axis "
                        "of the array.");
                goto fail;
            }
            PyArray_DIMS(self)[axis] = newdim / newtype->elsize;
            PyArray_STRIDES(self)[axis] = newtype->elsize;
        }
    }

    /* Viewing as a subarray increases the number of dimensions */
    if (PyDataType_HASSUBARRAY(newtype)) {
        /*
         * create new array object from data and update
         * dimensions, strides and descr from it
         */
        PyArrayObject *temp;
        /*
         * We would decref newtype here.
         * temp will steal a reference to it
         */
        temp = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, newtype, PyArray_NDIM(self),
                                 PyArray_DIMS(self), PyArray_STRIDES(self),
                                 PyArray_DATA(self), PyArray_FLAGS(self), NULL);
        if (temp == NULL) {
            return -1;
        }
        npy_free_cache_dim_array(self);
        ((PyArrayObject_fields *)self)->dimensions = PyArray_DIMS(temp);
        ((PyArrayObject_fields *)self)->nd = PyArray_NDIM(temp);
        ((PyArrayObject_fields *)self)->strides = PyArray_STRIDES(temp);
        newtype = PyArray_DESCR(temp);
        Py_INCREF(PyArray_DESCR(temp));
        /* Fool deallocator not to delete these*/
        ((PyArrayObject_fields *)temp)->nd = 0;
        ((PyArrayObject_fields *)temp)->dimensions = NULL;
        Py_DECREF(temp);
    }

    Py_DECREF(PyArray_DESCR(self));
    ((PyArrayObject_fields *)self)->descr = newtype;
    PyArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);
    return 0;

 fail:
    Py_DECREF(newtype);
    return -1;
}

static PyObject *
array_struct_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyArrayInterface *inter;

    inter = (PyArrayInterface *)PyArray_malloc(sizeof(PyArrayInterface));
    if (inter==NULL) {
        return PyErr_NoMemory();
    }
    inter->two = 2;
    inter->nd = PyArray_NDIM(self);
    inter->typekind = PyArray_DESCR(self)->kind;
    inter->itemsize = PyArray_DESCR(self)->elsize;
    inter->flags = PyArray_FLAGS(self);
    if (inter->flags & NPY_ARRAY_WARN_ON_WRITE) {
        /* Export a warn-on-write array as read-only */
        inter->flags = inter->flags & ~NPY_ARRAY_WARN_ON_WRITE;
        inter->flags = inter->flags & ~NPY_ARRAY_WRITEABLE;
    }
    /* reset unused flags */
    inter->flags &= ~(NPY_ARRAY_WRITEBACKIFCOPY | NPY_ARRAY_OWNDATA);
    if (PyArray_ISNOTSWAPPED(self)) inter->flags |= NPY_ARRAY_NOTSWAPPED;
    /*
     * Copy shape and strides over since these can be reset
     *when the array is "reshaped".
     */
    if (PyArray_NDIM(self) > 0) {
        inter->shape = (npy_intp *)PyArray_malloc(2*sizeof(npy_intp)*PyArray_NDIM(self));
        if (inter->shape == NULL) {
            PyArray_free(inter);
            return PyErr_NoMemory();
        }
        inter->strides = inter->shape + PyArray_NDIM(self);
        if (PyArray_NDIM(self)) {
            memcpy(inter->shape, PyArray_DIMS(self), sizeof(npy_intp)*PyArray_NDIM(self));
            memcpy(inter->strides, PyArray_STRIDES(self), sizeof(npy_intp)*PyArray_NDIM(self));
        }
    }
    else {
        inter->shape = NULL;
        inter->strides = NULL;
    }
    inter->data = PyArray_DATA(self);
    if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
        inter->descr = arraydescr_protocol_descr_get(PyArray_DESCR(self), NULL);
        if (inter->descr == NULL) {
            PyErr_Clear();
        }
        else {
            inter->flags &= NPY_ARR_HAS_DESCR;
        }
    }
    else {
        inter->descr = NULL;
    }
    PyObject *ret = PyCapsule_New(inter, NULL, gentype_struct_free);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyCapsule_SetContext(ret, self) < 0) {
        return NULL;
    }
    return ret;
}

static PyObject *
array_base_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    if (PyArray_BASE(self) == NULL) {
        Py_RETURN_NONE;
    }
    else {
        Py_INCREF(PyArray_BASE(self));
        return PyArray_BASE(self);
    }
}

/*
 * Create a view of a complex array with an equivalent data-type
 * except it is real instead of complex.
 */
static PyArrayObject *
_get_part(PyArrayObject *self, int imag)
{
    int float_type_num;
    PyArray_Descr *type;
    PyArrayObject *ret;
    int offset;

    switch (PyArray_DESCR(self)->type_num) {
        case NPY_CFLOAT:
            float_type_num = NPY_FLOAT;
            break;
        case NPY_CDOUBLE:
            float_type_num = NPY_DOUBLE;
            break;
        case NPY_CLONGDOUBLE:
            float_type_num = NPY_LONGDOUBLE;
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                     "Cannot convert complex type number %d to float",
                     PyArray_DESCR(self)->type_num);
            return NULL;

    }
    type = PyArray_DescrFromType(float_type_num);
    if (type == NULL) {
        return NULL;
    }

    offset = (imag ? type->elsize : 0);

    if (!PyArray_ISNBO(PyArray_DESCR(self)->byteorder)) {
        Py_SETREF(type, PyArray_DescrNew(type));
        if (type == NULL) {
            return NULL;
        }
        type->byteorder = PyArray_DESCR(self)->byteorder;
    }
    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            Py_TYPE(self),
            type,
            PyArray_NDIM(self),
            PyArray_DIMS(self),
            PyArray_STRIDES(self),
            PyArray_BYTES(self) + offset,
            PyArray_FLAGS(self), (PyObject *)self, (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

/* For Object arrays, we need to get and set the
   real part of each element.
 */

static PyObject *
array_real_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        return (PyObject *)ret;
    }
    else {
        Py_INCREF(self);
        return (PyObject *)self;
    }
}


static int
array_real_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;
    PyArrayObject *new;
    int retcode;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array real part");
        return -1;
    }
    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        if (ret == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(self);
        ret = self;
    }
    new = (PyArrayObject *)PyArray_FROM_O(val);
    if (new == NULL) {
        Py_DECREF(ret);
        return -1;
    }
    retcode = PyArray_MoveInto(ret, new);
    Py_DECREF(ret);
    Py_DECREF(new);
    return retcode;
}

/* For Object arrays we need to get
   and set the imaginary part of
   each element
*/

static PyObject *
array_imag_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 1);
    }
    else {
        Py_INCREF(PyArray_DESCR(self));
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                Py_TYPE(self),
                PyArray_DESCR(self),
                PyArray_NDIM(self),
                PyArray_DIMS(self),
                NULL, NULL,
                PyArray_ISFORTRAN(self),
                (PyObject *)self, NULL, _NPY_ARRAY_ZEROED);
        if (ret == NULL) {
            return NULL;
        }
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return (PyObject *) ret;
}

static int
array_imag_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array imaginary part");
        return -1;
    }
    if (PyArray_ISCOMPLEX(self)) {
        PyArrayObject *ret;
        PyArrayObject *new;
        int retcode;

        ret = _get_part(self, 1);
        if (ret == NULL) {
            return -1;
        }
        new = (PyArrayObject *)PyArray_FROM_O(val);
        if (new == NULL) {
            Py_DECREF(ret);
            return -1;
        }
        retcode = PyArray_MoveInto(ret, new);
        Py_DECREF(ret);
        Py_DECREF(new);
        return retcode;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "array does not have imaginary part to set");
        return -1;
    }
}

static PyObject *
array_flat_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IterNew((PyObject *)self);
}

static int
array_flat_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    PyArrayObject *arr = NULL;
    int retval = -1;
    PyArrayIterObject *selfit = NULL, *arrit = NULL;
    PyArray_Descr *typecode;
    int swap;
    PyArray_CopySwapFunc *copyswap;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array flat iterator");
        return -1;
    }
    if (PyArray_FailUnlessWriteable(self, "array") < 0) return -1;
    typecode = PyArray_DESCR(self);
    Py_INCREF(typecode);
    arr = (PyArrayObject *)PyArray_FromAny(val, typecode,
                  0, 0, NPY_ARRAY_FORCECAST | PyArray_FORTRAN_IF(self), NULL);
    if (arr == NULL) {
        return -1;
    }
    arrit = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (arrit == NULL) {
        goto exit;
    }
    selfit = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (selfit == NULL) {
        goto exit;
    }
    if (arrit->size == 0) {
        retval = 0;
        goto exit;
    }
    swap = PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(arr);
    copyswap = PyArray_DESCR(self)->f->copyswap;
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        while (selfit->index < selfit->size) {
            PyArray_Item_XDECREF(selfit->dataptr, PyArray_DESCR(self));
            PyArray_Item_INCREF(arrit->dataptr, PyArray_DESCR(arr));
            memmove(selfit->dataptr, arrit->dataptr, sizeof(PyObject **));
            if (swap) {
                copyswap(selfit->dataptr, NULL, swap, self);
            }
            PyArray_ITER_NEXT(selfit);
            PyArray_ITER_NEXT(arrit);
            if (arrit->index == arrit->size) {
                PyArray_ITER_RESET(arrit);
            }
        }
        retval = 0;
        goto exit;
    }

    while(selfit->index < selfit->size) {
        copyswap(selfit->dataptr, arrit->dataptr, swap, self);
        PyArray_ITER_NEXT(selfit);
        PyArray_ITER_NEXT(arrit);
        if (arrit->index == arrit->size) {
            PyArray_ITER_RESET(arrit);
        }
    }
    retval = 0;

 exit:
    Py_XDECREF(selfit);
    Py_XDECREF(arrit);
    Py_XDECREF(arr);
    return retval;
}

static PyObject *
array_transpose_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_Transpose(self, NULL);
}

NPY_NO_EXPORT PyGetSetDef array_getsetlist[] = {
    {"ndim",
        (getter)array_ndim_get,
        NULL,
        NULL, NULL},
    {"flags",
        (getter)array_flags_get,
        NULL,
        NULL, NULL},
    {"shape",
        (getter)array_shape_get,
        (setter)array_shape_set,
        NULL, NULL},
    {"strides",
        (getter)array_strides_get,
        (setter)array_strides_set,
        NULL, NULL},
    {"data",
        (getter)array_data_get,
        (setter)array_data_set,
        NULL, NULL},
    {"itemsize",
        (getter)array_itemsize_get,
        NULL,
        NULL, NULL},
    {"size",
        (getter)array_size_get,
        NULL,
        NULL, NULL},
    {"nbytes",
        (getter)array_nbytes_get,
        NULL,
        NULL, NULL},
    {"base",
        (getter)array_base_get,
        NULL,
        NULL, NULL},
    {"dtype",
        (getter)array_descr_get,
        (setter)array_descr_set,
        NULL, NULL},
    {"real",
        (getter)array_real_get,
        (setter)array_real_set,
        NULL, NULL},
    {"imag",
        (getter)array_imag_get,
        (setter)array_imag_set,
        NULL, NULL},
    {"flat",
        (getter)array_flat_get,
        (setter)array_flat_set,
        NULL, NULL},
    {"ctypes",
        (getter)array_ctypes_get,
        NULL,
        NULL, NULL},
    {"T",
        (getter)array_transpose_get,
        NULL,
        NULL, NULL},
    {"__array_interface__",
        (getter)array_interface_get,
        NULL,
        NULL, NULL},
    {"__array_struct__",
        (getter)array_struct_get,
        NULL,
        NULL, NULL},
    {"__array_priority__",
        (getter)array_priority_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};

/****************** end of attribute get and set routines *******************/

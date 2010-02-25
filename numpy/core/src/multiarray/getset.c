/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

#include "common.h"
#include "scalartypes.h"
#include "descriptor.h"
#include "getset.h"

/*******************  array attribute get and set routines ******************/

static PyObject *
array_ndim_get(PyArrayObject *self)
{
    return PyInt_FromLong(self->nd);
}

static PyObject *
array_flags_get(PyArrayObject *self)
{
    return PyArray_NewFlagsObject((PyObject *)self);
}

static PyObject *
array_shape_get(PyArrayObject *self)
{
    return PyArray_IntTupleFromIntp(self->nd, self->dimensions);
}


static int
array_shape_set(PyArrayObject *self, PyObject *val)
{
    int nd;
    PyObject *ret;

    /* Assumes C-order */
    ret = PyArray_Reshape(self, val);
    if (ret == NULL) {
        return -1;
    }
    if (PyArray_DATA(ret) != PyArray_DATA(self)) {
        Py_DECREF(ret);
        PyErr_SetString(PyExc_AttributeError,
                        "incompatible shape for a non-contiguous "\
                        "array");
        return -1;
    }

    /* Free old dimensions and strides */
    PyDimMem_FREE(self->dimensions);
    nd = PyArray_NDIM(ret);
    self->nd = nd;
    if (nd > 0) {
        /* create new dimensions and strides */
        self->dimensions = PyDimMem_NEW(2*nd);
        if (self->dimensions == NULL) {
            Py_DECREF(ret);
            PyErr_SetString(PyExc_MemoryError,"");
            return -1;
        }
        self->strides = self->dimensions + nd;
        memcpy(self->dimensions, PyArray_DIMS(ret), nd*sizeof(intp));
        memcpy(self->strides, PyArray_STRIDES(ret), nd*sizeof(intp));
    }
    else {
        self->dimensions = NULL;
        self->strides = NULL;
    }
    Py_DECREF(ret);
    PyArray_UpdateFlags(self, CONTIGUOUS | FORTRAN);
    return 0;
}


static PyObject *
array_strides_get(PyArrayObject *self)
{
    return PyArray_IntTupleFromIntp(self->nd, self->strides);
}

static int
array_strides_set(PyArrayObject *self, PyObject *obj)
{
    PyArray_Dims newstrides = {NULL, 0};
    PyArrayObject *new;
    intp numbytes = 0;
    intp offset = 0;
    Py_ssize_t buf_len;
    char *buf;

    if (!PyArray_IntpConverter(obj, &newstrides) ||
        newstrides.ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid strides");
        return -1;
    }
    if (newstrides.len != self->nd) {
        PyErr_Format(PyExc_ValueError, "strides must be "       \
                     " same length as shape (%d)", self->nd);
        goto fail;
    }
    new = self;
    while(new->base && PyArray_Check(new->base)) {
        new = (PyArrayObject *)(new->base);
    }
    /*
     * Get the available memory through the buffer interface on
     * new->base or if that fails from the current new
     */
    if (new->base && PyObject_AsReadBuffer(new->base,
                                           (const void **)&buf,
                                           &buf_len) >= 0) {
        offset = self->data - buf;
        numbytes = buf_len + offset;
    }
    else {
        PyErr_Clear();
        numbytes = PyArray_MultiplyList(new->dimensions,
                                        new->nd)*new->descr->elsize;
        offset = self->data - new->data;
    }

    if (!PyArray_CheckStrides(self->descr->elsize, self->nd, numbytes,
                              offset,
                              self->dimensions, newstrides.ptr)) {
        PyErr_SetString(PyExc_ValueError, "strides is not "\
                        "compatible with available memory");
        goto fail;
    }
    memcpy(self->strides, newstrides.ptr, sizeof(intp)*newstrides.len);
    PyArray_UpdateFlags(self, CONTIGUOUS | FORTRAN);
    PyDimMem_FREE(newstrides.ptr);
    return 0;

 fail:
    PyDimMem_FREE(newstrides.ptr);
    return -1;
}



static PyObject *
array_priority_get(PyArrayObject *self)
{
    if (PyArray_CheckExact(self)) {
        return PyFloat_FromDouble(PyArray_PRIORITY);
    }
    else {
        return PyFloat_FromDouble(PyArray_SUBTYPE_PRIORITY);
    }
}

static PyObject *
array_typestr_get(PyArrayObject *self)
{
    return arraydescr_protocol_typestr_get(self->descr);
}

static PyObject *
array_descr_get(PyArrayObject *self)
{
    Py_INCREF(self->descr);
    return (PyObject *)self->descr;
}

static PyObject *
array_protocol_descr_get(PyArrayObject *self)
{
    PyObject *res;
    PyObject *dobj;

    res = arraydescr_protocol_descr_get(self->descr);
    if (res) {
        return res;
    }
    PyErr_Clear();

    /* get default */
    dobj = PyTuple_New(2);
    if (dobj == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(dobj, 0, PyString_FromString(""));
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
    if PyArray_ISCONTIGUOUS(self) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyArray_IntTupleFromIntp(self->nd, self->strides);
}



static PyObject *
array_dataptr_get(PyArrayObject *self)
{
    return Py_BuildValue("NO",
                         PyLong_FromVoidPtr(self->data),
                         (self->flags & WRITEABLE ? Py_False :
                          Py_True));
}

static PyObject *
array_ctypes_get(PyArrayObject *self)
{
    PyObject *_numpy_internal;
    PyObject *ret;
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(_numpy_internal, "_ctypes", "ON", self,
                              PyLong_FromVoidPtr(self->data));
    Py_DECREF(_numpy_internal);
    return ret;
}

static PyObject *
array_interface_get(PyArrayObject *self)
{
    PyObject *dict;
    PyObject *obj;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    /* dataptr */
    obj = array_dataptr_get(self);
    PyDict_SetItemString(dict, "data", obj);
    Py_DECREF(obj);

    obj = array_protocol_strides_get(self);
    PyDict_SetItemString(dict, "strides", obj);
    Py_DECREF(obj);

    obj = array_protocol_descr_get(self);
    PyDict_SetItemString(dict, "descr", obj);
    Py_DECREF(obj);

    obj = arraydescr_protocol_typestr_get(self->descr);
    PyDict_SetItemString(dict, "typestr", obj);
    Py_DECREF(obj);

    obj = array_shape_get(self);
    PyDict_SetItemString(dict, "shape", obj);
    Py_DECREF(obj);

    obj = PyInt_FromLong(3);
    PyDict_SetItemString(dict, "version", obj);
    Py_DECREF(obj);

    return dict;
}

static PyObject *
array_data_get(PyArrayObject *self)
{
#if defined(NPY_PY3K)
    return PyMemoryView_FromObject(self);
#else
    intp nbytes;
    if (!(PyArray_ISONESEGMENT(self))) {
        PyErr_SetString(PyExc_AttributeError, "cannot get single-"\
                        "segment buffer for discontiguous array");
        return NULL;
    }
    nbytes = PyArray_NBYTES(self);
    if (PyArray_ISWRITEABLE(self)) {
        return PyBuffer_FromReadWriteObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
    else {
        return PyBuffer_FromObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
#endif
}

static int
array_data_set(PyArrayObject *self, PyObject *op)
{
    void *buf;
    Py_ssize_t buf_len;
    int writeable=1;

    if (PyObject_AsWriteBuffer(op, &buf, &buf_len) < 0) {
        writeable = 0;
        if (PyObject_AsReadBuffer(op, (const void **)&buf, &buf_len) < 0) {
            PyErr_SetString(PyExc_AttributeError,
                            "object does not have single-segment " \
                            "buffer interface");
            return -1;
        }
    }
    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_AttributeError, "cannot set single-" \
                        "segment buffer for discontiguous array");
        return -1;
    }
    if (PyArray_NBYTES(self) > buf_len) {
        PyErr_SetString(PyExc_AttributeError, "not enough data for array");
        return -1;
    }
    if (self->flags & OWNDATA) {
        PyArray_XDECREF(self);
        PyDataMem_FREE(self->data);
    }
    if (self->base) {
        if (self->flags & UPDATEIFCOPY) {
            ((PyArrayObject *)self->base)->flags |= WRITEABLE;
            self->flags &= ~UPDATEIFCOPY;
        }
        Py_DECREF(self->base);
    }
    Py_INCREF(op);
    self->base = op;
    self->data = buf;
    self->flags = CARRAY;
    if (!writeable) {
        self->flags &= ~WRITEABLE;
    }
    return 0;
}


static PyObject *
array_itemsize_get(PyArrayObject *self)
{
    return PyInt_FromLong((long) self->descr->elsize);
}

static PyObject *
array_size_get(PyArrayObject *self)
{
    intp size=PyArray_SIZE(self);
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) size);
#else
    if (size > MAX_LONG || size < MIN_LONG) {
        return PyLong_FromLongLong(size);
    }
    else {
        return PyInt_FromLong((long) size);
    }
#endif
}

static PyObject *
array_nbytes_get(PyArrayObject *self)
{
    intp nbytes = PyArray_NBYTES(self);
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) nbytes);
#else
    if (nbytes > MAX_LONG || nbytes < MIN_LONG) {
        return PyLong_FromLongLong(nbytes);
    }
    else {
        return PyInt_FromLong((long) nbytes);
    }
#endif
}


/*
 * If the type is changed.
 * Also needing change: strides, itemsize
 *
 * Either itemsize is exactly the same or the array is single-segment
 * (contiguous or fortran) with compatibile dimensions The shape and strides
 * will be adjusted in that case as well.
 */

static int
array_descr_set(PyArrayObject *self, PyObject *arg)
{
    PyArray_Descr *newtype = NULL;
    intp newdim;
    int index;
    char *msg = "new type not compatible with array.";

    if (!(PyArray_DescrConverter(arg, &newtype)) ||
        newtype == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid data-type for array");
        return -1;
    }
    if (PyDataType_FLAGCHK(newtype, NPY_ITEM_HASOBJECT) ||
        PyDataType_FLAGCHK(newtype, NPY_ITEM_IS_POINTER) ||
        PyDataType_FLAGCHK(self->descr, NPY_ITEM_HASOBJECT) ||
        PyDataType_FLAGCHK(self->descr, NPY_ITEM_IS_POINTER)) {
        PyErr_SetString(PyExc_TypeError,                      \
                        "Cannot change data-type for object " \
                        "array.");
        Py_DECREF(newtype);
        return -1;
    }

    if (newtype->elsize == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "data-type must not be 0-sized");
        Py_DECREF(newtype);
        return -1;
    }


    if ((newtype->elsize != self->descr->elsize) &&
        (self->nd == 0 || !PyArray_ISONESEGMENT(self) ||
         newtype->subarray)) {
        goto fail;
    }
    if (PyArray_ISCONTIGUOUS(self)) {
        index = self->nd - 1;
    }
    else {
        index = 0;
    }
    if (newtype->elsize < self->descr->elsize) {
        /*
         * if it is compatible increase the size of the
         * dimension at end (or at the front for FORTRAN)
         */
        if (self->descr->elsize % newtype->elsize != 0) {
            goto fail;
        }
        newdim = self->descr->elsize / newtype->elsize;
        self->dimensions[index] *= newdim;
        self->strides[index] = newtype->elsize;
    }
    else if (newtype->elsize > self->descr->elsize) {
        /*
         * Determine if last (or first if FORTRAN) dimension
         * is compatible
         */
        newdim = self->dimensions[index] * self->descr->elsize;
        if ((newdim % newtype->elsize) != 0) {
            goto fail;
        }
        self->dimensions[index] = newdim / newtype->elsize;
        self->strides[index] = newtype->elsize;
    }

    /* fall through -- adjust type*/
    Py_DECREF(self->descr);
    if (newtype->subarray) {
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
            PyArray_NewFromDescr(&PyArray_Type, newtype, self->nd,
                                 self->dimensions, self->strides,
                                 self->data, self->flags, NULL);
        if (temp == NULL) {
            return -1;
        }
        PyDimMem_FREE(self->dimensions);
        self->dimensions = temp->dimensions;
        self->nd = temp->nd;
        self->strides = temp->strides;
        newtype = temp->descr;
        Py_INCREF(temp->descr);
        /* Fool deallocator not to delete these*/
        temp->nd = 0;
        temp->dimensions = NULL;
        Py_DECREF(temp);
    }

    self->descr = newtype;
    PyArray_UpdateFlags(self, UPDATE_ALL);
    return 0;

 fail:
    PyErr_SetString(PyExc_ValueError, msg);
    Py_DECREF(newtype);
    return -1;
}

static PyObject *
array_struct_get(PyArrayObject *self)
{
    PyArrayInterface *inter;
    PyObject *ret;

    inter = (PyArrayInterface *)_pya_malloc(sizeof(PyArrayInterface));
    if (inter==NULL) {
        return PyErr_NoMemory();
    }
    inter->two = 2;
    inter->nd = self->nd;
    inter->typekind = self->descr->kind;
    inter->itemsize = self->descr->elsize;
    inter->flags = self->flags;
    /* reset unused flags */
    inter->flags &= ~(UPDATEIFCOPY | OWNDATA);
    if (PyArray_ISNOTSWAPPED(self)) inter->flags |= NOTSWAPPED;
    /*
     * Copy shape and strides over since these can be reset
     *when the array is "reshaped".
     */
    if (self->nd > 0) {
        inter->shape = (intp *)_pya_malloc(2*sizeof(intp)*self->nd);
        if (inter->shape == NULL) {
            _pya_free(inter);
            return PyErr_NoMemory();
        }
        inter->strides = inter->shape + self->nd;
        memcpy(inter->shape, self->dimensions, sizeof(intp)*self->nd);
        memcpy(inter->strides, self->strides, sizeof(intp)*self->nd);
    }
    else {
        inter->shape = NULL;
        inter->strides = NULL;
    }
    inter->data = self->data;
    if (self->descr->names) {
        inter->descr = arraydescr_protocol_descr_get(self->descr);
        if (inter->descr == NULL) {
            PyErr_Clear();
        }
        else {
            inter->flags &= ARR_HAS_DESCR;
        }
    }
    else {
        inter->descr = NULL;
    }
    Py_INCREF(self);
#if defined(NPY_PY3K)
    ret = PyCapsule_New(inter, NULL, gentype_struct_free);
    if (ret == NULL) {
        PyErr_Clear();
    }
    else if (PyCapsule_SetContext(ret, self) != 0) {
        PyErr_Clear();
        Py_DECREF(ret);
        ret = NULL;
    }
#else
    ret = PyCObject_FromVoidPtrAndDesc(inter, self, gentype_struct_free);
#endif
    return ret;
}

static PyObject *
array_base_get(PyArrayObject *self)
{
    if (self->base == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else {
        Py_INCREF(self->base);
        return self->base;
    }
}

/*
 * Create a view of a complex array with an equivalent data-type
 * except it is real instead of complex.
 */
static PyArrayObject *
_get_part(PyArrayObject *self, int imag)
{
    PyArray_Descr *type;
    PyArrayObject *ret;
    int offset;

    type = PyArray_DescrFromType(self->descr->type_num -
                                 PyArray_NUM_FLOATTYPE);
    offset = (imag ? type->elsize : 0);

    if (!PyArray_ISNBO(self->descr->byteorder)) {
        PyArray_Descr *new;
        new = PyArray_DescrNew(type);
        new->byteorder = self->descr->byteorder;
        Py_DECREF(type);
        type = new;
    }
    ret = (PyArrayObject *)
        PyArray_NewFromDescr(Py_TYPE(self),
                             type,
                             self->nd,
                             self->dimensions,
                             self->strides,
                             self->data + offset,
                             self->flags, (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    ret->flags &= ~CONTIGUOUS;
    ret->flags &= ~FORTRAN;
    Py_INCREF(self);
    ret->base = (PyObject *)self;
    return ret;
}

/* For Object arrays, we need to get and set the
   real part of each element.
 */

static PyObject *
array_real_get(PyArrayObject *self)
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
array_real_set(PyArrayObject *self, PyObject *val)
{
    PyArrayObject *ret;
    PyArrayObject *new;
    int rint;

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
    new = (PyArrayObject *)PyArray_FromAny(val, NULL, 0, 0, 0, NULL);
    if (new == NULL) {
        Py_DECREF(ret);
        return -1;
    }
    rint = PyArray_MoveInto(ret, new);
    Py_DECREF(ret);
    Py_DECREF(new);
    return rint;
}

/* For Object arrays we need to get
   and set the imaginary part of
   each element
*/

static PyObject *
array_imag_get(PyArrayObject *self)
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 1);
    }
    else {
        Py_INCREF(self->descr);
        ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                                    self->descr,
                                                    self->nd,
                                                    self->dimensions,
                                                    NULL, NULL,
                                                    PyArray_ISFORTRAN(self),
                                                    (PyObject *)self);
        if (ret == NULL) {
            return NULL;
        }
        if (_zerofill(ret) < 0) {
            return NULL;
        }
        ret->flags &= ~WRITEABLE;
    }
    return (PyObject *) ret;
}

static int
array_imag_set(PyArrayObject *self, PyObject *val)
{
    if (PyArray_ISCOMPLEX(self)) {
        PyArrayObject *ret;
        PyArrayObject *new;
        int rint;

        ret = _get_part(self, 1);
        if (ret == NULL) {
            return -1;
        }
        new = (PyArrayObject *)PyArray_FromAny(val, NULL, 0, 0, 0, NULL);
        if (new == NULL) {
            Py_DECREF(ret);
            return -1;
        }
        rint = PyArray_MoveInto(ret, new);
        Py_DECREF(ret);
        Py_DECREF(new);
        return rint;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "array does not have "\
                        "imaginary part to set");
        return -1;
    }
}

static PyObject *
array_flat_get(PyArrayObject *self)
{
    return PyArray_IterNew((PyObject *)self);
}

static int
array_flat_set(PyArrayObject *self, PyObject *val)
{
    PyObject *arr = NULL;
    int retval = -1;
    PyArrayIterObject *selfit = NULL, *arrit = NULL;
    PyArray_Descr *typecode;
    int swap;
    PyArray_CopySwapFunc *copyswap;

    typecode = self->descr;
    Py_INCREF(typecode);
    arr = PyArray_FromAny(val, typecode,
                          0, 0, FORCECAST | FORTRAN_IF(self), NULL);
    if (arr == NULL) {
        return -1;
    }
    arrit = (PyArrayIterObject *)PyArray_IterNew(arr);
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
    copyswap = self->descr->f->copyswap;
    if (PyDataType_REFCHK(self->descr)) {
        while (selfit->index < selfit->size) {
            PyArray_Item_XDECREF(selfit->dataptr, self->descr);
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
        memmove(selfit->dataptr, arrit->dataptr, self->descr->elsize);
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

 exit:
    Py_XDECREF(selfit);
    Py_XDECREF(arrit);
    Py_XDECREF(arr);
    return retval;
}

static PyObject *
array_transpose_get(PyArrayObject *self)
{
    return PyArray_Transpose(self, NULL);
}

/* If this is None, no function call is made
   --- default sub-class behavior
*/
static PyObject *
array_finalize_get(PyArrayObject *NPY_UNUSED(self))
{
    Py_INCREF(Py_None);
    return Py_None;
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
    {   "T",
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
    {"__array_finalize__",
        (getter)array_finalize_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};

/****************** end of attribute get and set routines *******************/

/* Array Flags Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "config.h"

#include "common.h"

static int
_IsContiguous(PyArrayObject *ap);

static int
_IsFortranContiguous(PyArrayObject *ap);

/*NUMPY_API
 *
 * Get New ArrayFlagsObject
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFlagsObject(PyObject *obj)
{
    PyObject *flagobj;
    int flags;
    if (obj == NULL) {
        flags = CONTIGUOUS | OWNDATA | FORTRAN | ALIGNED;
    }
    else {
        flags = PyArray_FLAGS(obj);
    }
    flagobj = PyArrayFlags_Type.tp_alloc(&PyArrayFlags_Type, 0);
    if (flagobj == NULL) {
        return NULL;
    }
    Py_XINCREF(obj);
    ((PyArrayFlagsObject *)flagobj)->arr = obj;
    ((PyArrayFlagsObject *)flagobj)->flags = flags;
    return flagobj;
}

/*NUMPY_API
 * Update Several Flags at once.
 */
NPY_NO_EXPORT void
PyArray_UpdateFlags(PyArrayObject *ret, int flagmask)
{

    if (flagmask & FORTRAN) {
        if (_IsFortranContiguous(ret)) {
            ret->flags |= FORTRAN;
            if (ret->nd > 1) {
                ret->flags &= ~CONTIGUOUS;
            }
        }
        else {
            ret->flags &= ~FORTRAN;
        }
    }
    if (flagmask & CONTIGUOUS) {
        if (_IsContiguous(ret)) {
            ret->flags |= CONTIGUOUS;
            if (ret->nd > 1) {
                ret->flags &= ~FORTRAN;
            }
        }
        else {
            ret->flags &= ~CONTIGUOUS;
        }
    }
    if (flagmask & ALIGNED) {
        if (_IsAligned(ret)) {
            ret->flags |= ALIGNED;
        }
        else {
            ret->flags &= ~ALIGNED;
        }
    }
    /*
     * This is not checked by default WRITEABLE is not
     * part of UPDATE_ALL
     */
    if (flagmask & WRITEABLE) {
        if (_IsWriteable(ret)) {
            ret->flags |= WRITEABLE;
        }
        else {
            ret->flags &= ~WRITEABLE;
        }
    }
    return;
}

/*
 * Check whether the given array is stored contiguously
 * (row-wise) in memory.
 *
 * 0-strided arrays are not contiguous (even if dimension == 1)
 */
static int
_IsContiguous(PyArrayObject *ap)
{
    intp sd;
    intp dim;
    int i;

    if (ap->nd == 0) {
        return 1;
    }
    sd = ap->descr->elsize;
    if (ap->nd == 1) {
        return ap->dimensions[0] == 1 || sd == ap->strides[0];
    }
    for (i = ap->nd - 1; i >= 0; --i) {
        dim = ap->dimensions[i];
        /* contiguous by definition */
        if (dim == 0) {
            return 1;
        }
        if (ap->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}


/* 0-strided arrays are not contiguous (even if dimension == 1) */
static int
_IsFortranContiguous(PyArrayObject *ap)
{
    intp sd;
    intp dim;
    int i;

    if (ap->nd == 0) {
        return 1;
    }
    sd = ap->descr->elsize;
    if (ap->nd == 1) {
        return ap->dimensions[0] == 1 || sd == ap->strides[0];
    }
    for (i = 0; i < ap->nd; ++i) {
        dim = ap->dimensions[i];
        /* fortran contiguous by definition */
        if (dim == 0) {
            return 1;
        }
        if (ap->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}

static void
arrayflags_dealloc(PyArrayFlagsObject *self)
{
    Py_XDECREF(self->arr);
    self->ob_type->tp_free((PyObject *)self);
}


#define _define_get(UPPER, lower)                                       \
    static PyObject *                                                   \
    arrayflags_ ## lower ## _get(PyArrayFlagsObject *self)              \
    {                                                                   \
        PyObject *item;                                                 \
        item = ((self->flags & (UPPER)) == (UPPER)) ? Py_True : Py_False; \
        Py_INCREF(item);                                                \
        return item;                                                    \
    }

_define_get(CONTIGUOUS, contiguous)
_define_get(FORTRAN, fortran)
_define_get(UPDATEIFCOPY, updateifcopy)
_define_get(OWNDATA, owndata)
_define_get(ALIGNED, aligned)
_define_get(WRITEABLE, writeable)

_define_get(ALIGNED|WRITEABLE, behaved)
_define_get(ALIGNED|WRITEABLE|CONTIGUOUS, carray)

static PyObject *
arrayflags_forc_get(PyArrayFlagsObject *self)
{
    PyObject *item;

    if (((self->flags & FORTRAN) == FORTRAN) ||
        ((self->flags & CONTIGUOUS) == CONTIGUOUS)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

static PyObject *
arrayflags_fnc_get(PyArrayFlagsObject *self)
{
    PyObject *item;

    if (((self->flags & FORTRAN) == FORTRAN) &&
        !((self->flags & CONTIGUOUS) == CONTIGUOUS)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

static PyObject *
arrayflags_farray_get(PyArrayFlagsObject *self)
{
    PyObject *item;

    if (((self->flags & (ALIGNED|WRITEABLE|FORTRAN)) ==
         (ALIGNED|WRITEABLE|FORTRAN)) &&
        !((self->flags & CONTIGUOUS) == CONTIGUOUS)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

static PyObject *
arrayflags_num_get(PyArrayFlagsObject *self)
{
    return PyInt_FromLong(self->flags);
}

/* relies on setflags order being write, align, uic */
static int
arrayflags_updateifcopy_set(PyArrayFlagsObject *self, PyObject *obj)
{
    PyObject *res;
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot set flags on array scalars.");
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None, Py_None,
                              (PyObject_IsTrue(obj) ? Py_True : Py_False));
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

static int
arrayflags_aligned_set(PyArrayFlagsObject *self, PyObject *obj)
{
    PyObject *res;
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot set flags on array scalars.");
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None,
                              (PyObject_IsTrue(obj) ? Py_True : Py_False),
                              Py_None);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

static int
arrayflags_writeable_set(PyArrayFlagsObject *self, PyObject *obj)
{
    PyObject *res;
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot set flags on array scalars.");
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO",
                              (PyObject_IsTrue(obj) ? Py_True : Py_False),
                              Py_None, Py_None);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}


static PyGetSetDef arrayflags_getsets[] = {
    {"contiguous",
        (getter)arrayflags_contiguous_get,
        NULL,
        "", NULL},
    {"c_contiguous",
        (getter)arrayflags_contiguous_get,
        NULL,
        "", NULL},
    {"f_contiguous",
        (getter)arrayflags_fortran_get,
        NULL,
        "", NULL},
    {"fortran",
        (getter)arrayflags_fortran_get,
        NULL,
        "", NULL},
    {"updateifcopy",
        (getter)arrayflags_updateifcopy_get,
        (setter)arrayflags_updateifcopy_set,
        "", NULL},
    {"owndata",
        (getter)arrayflags_owndata_get,
        NULL,
        "", NULL},
    {"aligned",
        (getter)arrayflags_aligned_get,
        (setter)arrayflags_aligned_set,
        "", NULL},
    {"writeable",
        (getter)arrayflags_writeable_get,
        (setter)arrayflags_writeable_set,
        "", NULL},
    {"fnc",
        (getter)arrayflags_fnc_get,
        NULL,
        "", NULL},
    {"forc",
        (getter)arrayflags_forc_get,
        NULL,
        "", NULL},
    {"behaved",
        (getter)arrayflags_behaved_get,
        NULL,
        "", NULL},
    {"carray",
        (getter)arrayflags_carray_get,
        NULL,
        "", NULL},
    {"farray",
        (getter)arrayflags_farray_get,
        NULL,
        "", NULL},
    {"num",
        (getter)arrayflags_num_get,
        NULL,
        "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *
arrayflags_getitem(PyArrayFlagsObject *self, PyObject *ind)
{
    char *key;
    int n;
    if (!PyString_Check(ind)) {
        goto fail;
    }
    key = PyString_AS_STRING(ind);
    n = PyString_GET_SIZE(ind);
    switch(n) {
    case 1:
        switch(key[0]) {
        case 'C':
            return arrayflags_contiguous_get(self);
        case 'F':
            return arrayflags_fortran_get(self);
        case 'W':
            return arrayflags_writeable_get(self);
        case 'B':
            return arrayflags_behaved_get(self);
        case 'O':
            return arrayflags_owndata_get(self);
        case 'A':
            return arrayflags_aligned_get(self);
        case 'U':
            return arrayflags_updateifcopy_get(self);
        default:
            goto fail;
        }
        break;
    case 2:
        if (strncmp(key, "CA", n) == 0) {
            return arrayflags_carray_get(self);
        }
        if (strncmp(key, "FA", n) == 0) {
            return arrayflags_farray_get(self);
        }
        break;
    case 3:
        if (strncmp(key, "FNC", n) == 0) {
            return arrayflags_fnc_get(self);
        }
        break;
    case 4:
        if (strncmp(key, "FORC", n) == 0) {
            return arrayflags_forc_get(self);
        }
        break;
    case 6:
        if (strncmp(key, "CARRAY", n) == 0) {
            return arrayflags_carray_get(self);
        }
        if (strncmp(key, "FARRAY", n) == 0) {
            return arrayflags_farray_get(self);
        }
        break;
    case 7:
        if (strncmp(key,"FORTRAN",n) == 0) {
            return arrayflags_fortran_get(self);
        }
        if (strncmp(key,"BEHAVED",n) == 0) {
            return arrayflags_behaved_get(self);
        }
        if (strncmp(key,"OWNDATA",n) == 0) {
            return arrayflags_owndata_get(self);
        }
        if (strncmp(key,"ALIGNED",n) == 0) {
            return arrayflags_aligned_get(self);
        }
        break;
    case 9:
        if (strncmp(key,"WRITEABLE",n) == 0) {
            return arrayflags_writeable_get(self);
        }
        break;
    case 10:
        if (strncmp(key,"CONTIGUOUS",n) == 0) {
            return arrayflags_contiguous_get(self);
        }
        break;
    case 12:
        if (strncmp(key, "UPDATEIFCOPY", n) == 0) {
            return arrayflags_updateifcopy_get(self);
        }
        if (strncmp(key, "C_CONTIGUOUS", n) == 0) {
            return arrayflags_contiguous_get(self);
        }
        if (strncmp(key, "F_CONTIGUOUS", n) == 0) {
            return arrayflags_fortran_get(self);
        }
        break;
    }

 fail:
    PyErr_SetString(PyExc_KeyError, "Unknown flag");
    return NULL;
}

static int
arrayflags_setitem(PyArrayFlagsObject *self, PyObject *ind, PyObject *item)
{
    char *key;
    int n;
    if (!PyString_Check(ind)) {
        goto fail;
    }
    key = PyString_AS_STRING(ind);
    n = PyString_GET_SIZE(ind);
    if (((n==9) && (strncmp(key, "WRITEABLE", n) == 0)) ||
        ((n==1) && (strncmp(key, "W", n) == 0))) {
        return arrayflags_writeable_set(self, item);
    }
    else if (((n==7) && (strncmp(key, "ALIGNED", n) == 0)) ||
             ((n==1) && (strncmp(key, "A", n) == 0))) {
        return arrayflags_aligned_set(self, item);
    }
    else if (((n==12) && (strncmp(key, "UPDATEIFCOPY", n) == 0)) ||
             ((n==1) && (strncmp(key, "U", n) == 0))) {
        return arrayflags_updateifcopy_set(self, item);
    }

 fail:
    PyErr_SetString(PyExc_KeyError, "Unknown flag");
    return -1;
}

static char *
_torf_(int flags, int val)
{
    if ((flags & val) == val) {
        return "True";
    }
    else {
        return "False";
    }
}

static PyObject *
arrayflags_print(PyArrayFlagsObject *self)
{
    int fl = self->flags;

    return PyString_FromFormat("  %s : %s\n  %s : %s\n  %s : %s\n"\
                               "  %s : %s\n  %s : %s\n  %s : %s",
                               "C_CONTIGUOUS", _torf_(fl, CONTIGUOUS),
                               "F_CONTIGUOUS", _torf_(fl, FORTRAN),
                               "OWNDATA", _torf_(fl, OWNDATA),
                               "WRITEABLE", _torf_(fl, WRITEABLE),
                               "ALIGNED", _torf_(fl, ALIGNED),
                               "UPDATEIFCOPY", _torf_(fl, UPDATEIFCOPY));
}


static int
arrayflags_compare(PyArrayFlagsObject *self, PyArrayFlagsObject *other)
{
    if (self->flags == other->flags) {
        return 0;
    }
    else if (self->flags < other->flags) {
        return -1;
    }
    else {
        return 1;
    }
}

static PyMappingMethods arrayflags_as_mapping = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)NULL,                       /*mp_length*/
#else
    (inquiry)NULL,                       /*mp_length*/
#endif
    (binaryfunc)arrayflags_getitem,      /*mp_subscript*/
    (objobjargproc)arrayflags_setitem,   /*mp_ass_subscript*/
};


static PyObject *
arrayflags_new(PyTypeObject *NPY_UNUSED(self), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyObject *arg=NULL;
    if (!PyArg_UnpackTuple(args, "flagsobj", 0, 1, &arg)) {
        return NULL;
    }
    if ((arg != NULL) && PyArray_Check(arg)) {
        return PyArray_NewFlagsObject(arg);
    }
    else {
        return PyArray_NewFlagsObject(NULL);
    }
}

NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.flagsobj",
    sizeof(PyArrayFlagsObject),
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arrayflags_dealloc,             /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    (cmpfunc)arrayflags_compare,                /* tp_compare */
#endif
    (reprfunc)arrayflags_print,                 /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    &arrayflags_as_mapping,                     /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)arrayflags_print,                 /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    arrayflags_getsets,                         /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    arrayflags_new,                             /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
};

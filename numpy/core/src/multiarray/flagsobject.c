/* Array Flags Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "array_assign.h"

#include "common.h"

static void
_UpdateContiguousFlags(PyArrayObject *ap);

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
        flags = NPY_ARRAY_C_CONTIGUOUS |
                NPY_ARRAY_OWNDATA |
                NPY_ARRAY_F_CONTIGUOUS |
                NPY_ARRAY_ALIGNED;
    }
    else {
        if (!PyArray_Check(obj)) {
            PyErr_SetString(PyExc_ValueError,
                    "Need a NumPy array to create a flags object");
            return NULL;
        }

        flags = PyArray_FLAGS((PyArrayObject *)obj);
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
    /* Always update both, as its not trivial to guess one from the other */
    if (flagmask & (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) {
        _UpdateContiguousFlags(ret);
    }
    if (flagmask & NPY_ARRAY_ALIGNED) {
        if (IsAligned(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
    }
    /*
     * This is not checked by default WRITEABLE is not
     * part of UPDATE_ALL
     */
    if (flagmask & NPY_ARRAY_WRITEABLE) {
        if (_IsWriteable(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
    }
    return;
}

/*
 * Check whether the given array is stored contiguously
 * in memory. And update the passed in ap flags appropriately.
 *
 * The traditional rule is that for an array to be flagged as C contiguous,
 * the following must hold:
 *
 * strides[-1] == itemsize
 * strides[i] == shape[i+1] * strides[i + 1]
 *
 * And for an array to be flagged as F contiguous, the obvious reversal:
 *
 * strides[0] == itemsize
 * strides[i] == shape[i - 1] * strides[i - 1]
 *
 * According to these rules, a 0- or 1-dimensional array is either both
 * C- and F-contiguous, or neither; and an array with 2+ dimensions
 * can be C- or F- contiguous, or neither, but not both. Though there
 * there are exceptions for arrays with zero or one item, in the first
 * case the check is relaxed up to and including the first dimension
 * with shape[i] == 0. In the second case `strides == itemsize` will
 * can be true for all dimensions and both flags are set.
 *
 * When NPY_RELAXED_STRIDES_CHECKING is set, we use a more accurate
 * definition of C- and F-contiguity, in which all 0-sized arrays are
 * contiguous (regardless of dimensionality), and if shape[i] == 1
 * then we ignore strides[i] (since it has no affect on memory layout).
 * With these new rules, it is possible for e.g. a 10x1 array to be both
 * C- and F-contiguous -- but, they break downstream code which assumes
 * that for contiguous arrays strides[-1] (resp. strides[0]) always
 * contains the itemsize.
 */
static void
_UpdateContiguousFlags(PyArrayObject *ap)
{
    npy_intp sd;
    npy_intp dim;
    int i;
    npy_bool is_c_contig = 1;

    sd = PyArray_ITEMSIZE(ap);
    for (i = PyArray_NDIM(ap) - 1; i >= 0; --i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        /* contiguous by definition */
        if (dim == 0) {
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                is_c_contig = 0;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            is_c_contig = 0;
            break;
         }
        /* contiguous, if it got this far */
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    if (is_c_contig) {
        PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }
    else {
        PyArray_CLEARFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }

    /* check if fortran contiguous */
    sd = PyArray_ITEMSIZE(ap);
    for (i = 0; i < PyArray_NDIM(ap); ++i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
                return;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
    return;
}

static void
arrayflags_dealloc(PyArrayFlagsObject *self)
{
    Py_XDECREF(self->arr);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


#define _define_get(UPPER, lower) \
    static PyObject * \
    arrayflags_ ## lower ## _get(PyArrayFlagsObject *self) \
    { \
        PyObject *item; \
        item = ((self->flags & (UPPER)) == (UPPER)) ? Py_True : Py_False; \
        Py_INCREF(item); \
        return item; \
    }

_define_get(NPY_ARRAY_C_CONTIGUOUS, contiguous)
_define_get(NPY_ARRAY_F_CONTIGUOUS, fortran)
_define_get(NPY_ARRAY_WRITEBACKIFCOPY, writebackifcopy)
_define_get(NPY_ARRAY_OWNDATA, owndata)
_define_get(NPY_ARRAY_ALIGNED, aligned)
_define_get(NPY_ARRAY_WRITEABLE, writeable)
_define_get(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE, behaved)
_define_get(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE|
            NPY_ARRAY_C_CONTIGUOUS, carray)

static PyObject *
arrayflags_updateifcopy_get(PyArrayFlagsObject *self)
{
    PyObject *item;
    /* 2017-Nov-10 1.14 */
    if(DEPRECATE("UPDATEIFCOPY deprecated, use WRITEBACKIFCOPY instead") < 0) {
        return NULL;
    }
    if ((self->flags & (NPY_ARRAY_UPDATEIFCOPY)) == (NPY_ARRAY_UPDATEIFCOPY)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}


static PyObject *
arrayflags_forc_get(PyArrayFlagsObject *self)
{
    PyObject *item;

    if (((self->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) ||
        ((self->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
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

    if (((self->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) &&
        !((self->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
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

    if (((self->flags & (NPY_ARRAY_ALIGNED|
                         NPY_ARRAY_WRITEABLE|
                         NPY_ARRAY_F_CONTIGUOUS)) != 0) &&
        !((self->flags & NPY_ARRAY_C_CONTIGUOUS) != 0)) {
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

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags updateifcopy attribute");
        return -1;
    }
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    /* 2017-Nov-10 1.14 */
    if(DEPRECATE("UPDATEIFCOPY deprecated, use WRITEBACKIFCOPY instead") < 0) {
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

/* relies on setflags order being write, align, uic */
static int
arrayflags_writebackifcopy_set(PyArrayFlagsObject *self, PyObject *obj)
{
    PyObject *res;

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags writebackifcopy attribute");
        return -1;
    }
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
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

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags aligned attribute");
        return -1;
    }
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
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

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags writeable attribute");
        return -1;
    }
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
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
        NULL, NULL},
    {"c_contiguous",
        (getter)arrayflags_contiguous_get,
        NULL,
        NULL, NULL},
    {"f_contiguous",
        (getter)arrayflags_fortran_get,
        NULL,
        NULL, NULL},
    {"fortran",
        (getter)arrayflags_fortran_get,
        NULL,
        NULL, NULL},
    {"updateifcopy",
        (getter)arrayflags_updateifcopy_get,
        (setter)arrayflags_updateifcopy_set,
        NULL, NULL},
    {"writebackifcopy",
        (getter)arrayflags_writebackifcopy_get,
        (setter)arrayflags_writebackifcopy_set,
        NULL, NULL},
    {"owndata",
        (getter)arrayflags_owndata_get,
        NULL,
        NULL, NULL},
    {"aligned",
        (getter)arrayflags_aligned_get,
        (setter)arrayflags_aligned_set,
        NULL, NULL},
    {"writeable",
        (getter)arrayflags_writeable_get,
        (setter)arrayflags_writeable_set,
        NULL, NULL},
    {"fnc",
        (getter)arrayflags_fnc_get,
        NULL,
        NULL, NULL},
    {"forc",
        (getter)arrayflags_forc_get,
        NULL,
        NULL, NULL},
    {"behaved",
        (getter)arrayflags_behaved_get,
        NULL,
        NULL, NULL},
    {"carray",
        (getter)arrayflags_carray_get,
        NULL,
        NULL, NULL},
    {"farray",
        (getter)arrayflags_farray_get,
        NULL,
        NULL, NULL},
    {"num",
        (getter)arrayflags_num_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *
arrayflags_getitem(PyArrayFlagsObject *self, PyObject *ind)
{
    char *key = NULL;
    char buf[16];
    int n;
    if (PyUnicode_Check(ind)) {
        PyObject *tmp_str;
        tmp_str = PyUnicode_AsASCIIString(ind);
        if (tmp_str == NULL) {
            return NULL;
        }
        key = PyBytes_AS_STRING(tmp_str);
        n = PyBytes_GET_SIZE(tmp_str);
        if (n > 16) {
            Py_DECREF(tmp_str);
            goto fail;
        }
        memcpy(buf, key, n);
        Py_DECREF(tmp_str);
        key = buf;
    }
    else if (PyBytes_Check(ind)) {
        key = PyBytes_AS_STRING(ind);
        n = PyBytes_GET_SIZE(ind);
    }
    else {
        goto fail;
    }
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
        case 'X':
            return arrayflags_writebackifcopy_get(self);
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
    case 15:
        if (strncmp(key, "WRITEBACKIFCOPY", n) == 0) {
            return arrayflags_writebackifcopy_get(self);
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
    char buf[16];
    int n;
    if (PyUnicode_Check(ind)) {
        PyObject *tmp_str;
        tmp_str = PyUnicode_AsASCIIString(ind);
        key = PyBytes_AS_STRING(tmp_str);
        n = PyBytes_GET_SIZE(tmp_str);
        if (n > 16) n = 16;
        memcpy(buf, key, n);
        Py_DECREF(tmp_str);
        key = buf;
    }
    else if (PyBytes_Check(ind)) {
        key = PyBytes_AS_STRING(ind);
        n = PyBytes_GET_SIZE(ind);
    }
    else {
        goto fail;
    }
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
    else if (((n==14) && (strncmp(key, "WRITEBACKIFCOPY", n) == 0)) ||
             ((n==1) && (strncmp(key, "X", n) == 0))) {
        return arrayflags_writebackifcopy_set(self, item);
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

    return PyUString_FromFormat(
                        "  %s : %s\n  %s : %s\n"
                        "  %s : %s\n  %s : %s\n"
                        "  %s : %s\n  %s : %s\n"
                        "  %s : %s",
                        "C_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_C_CONTIGUOUS),
                        "F_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_F_CONTIGUOUS),
                        "OWNDATA",         _torf_(fl, NPY_ARRAY_OWNDATA),
                        "WRITEABLE",       _torf_(fl, NPY_ARRAY_WRITEABLE),
                        "ALIGNED",         _torf_(fl, NPY_ARRAY_ALIGNED),
                        "WRITEBACKIFCOPY", _torf_(fl, NPY_ARRAY_WRITEBACKIFCOPY),
                        "UPDATEIFCOPY",    _torf_(fl, NPY_ARRAY_UPDATEIFCOPY));
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


static PyObject*
arrayflags_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *result = Py_NotImplemented;
    int cmp;

    if (cmp_op != Py_EQ && cmp_op != Py_NE) {
        PyErr_SetString(PyExc_TypeError,
                        "undefined comparison for flag object");
        return NULL;
    }

    if (PyObject_TypeCheck(other, &PyArrayFlags_Type)) {
        cmp = arrayflags_compare((PyArrayFlagsObject *)self,
                                 (PyArrayFlagsObject *)other);

        if (cmp_op == Py_EQ) {
            result = (cmp == 0) ? Py_True : Py_False;
        }
        else if (cmp_op == Py_NE) {
            result = (cmp != 0) ? Py_True : Py_False;
        }
    }

    Py_INCREF(result);
    return result;
}

static PyMappingMethods arrayflags_as_mapping = {
    (lenfunc)NULL,                       /*mp_length*/
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
    arrayflags_richcompare,                     /* tp_richcompare */
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
    0,                                          /* tp_version_tag */
};

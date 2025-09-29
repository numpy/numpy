/* Array Flags Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"


#include "array_assign.h"

#include "common.h"
#include "flagsobject.h"


static void
_UpdateContiguousFlags(PyArrayObject *ap);

/*
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
 * can be C- or F- contiguous, or neither, but not both (unless it has only
 * a single element).
 * We correct this, however.  When a dimension has length 1, its stride is
 * never used and thus has no effect on the  memory layout.
 * The above rules thus only apply when ignoring all size 1 dimensions.
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
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
                return;
            }
            sd *= dim;
        }
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
    arrayflags_ ## lower ## _get( \
            PyArrayFlagsObject *self, void *NPY_UNUSED(ignored)) \
    { \
        return PyBool_FromLong((self->flags & (UPPER)) == (UPPER)); \
    }

static char *msg = "future versions will not create a writeable "
    "array from broadcast_array. Set the writable flag explicitly to "
    "avoid this warning.";

#define _define_get_warn(UPPER, lower) \
    static PyObject * \
    arrayflags_ ## lower ## _get( \
            PyArrayFlagsObject *self, void *NPY_UNUSED(ignored)) \
    { \
        if (self->flags & NPY_ARRAY_WARN_ON_WRITE) { \
            if (PyErr_WarnEx(PyExc_FutureWarning, msg, 1) < 0) {\
                return NULL; \
            } \
        }\
        return PyBool_FromLong((self->flags & (UPPER)) == (UPPER)); \
    }


_define_get(NPY_ARRAY_C_CONTIGUOUS, contiguous)
_define_get(NPY_ARRAY_F_CONTIGUOUS, fortran)
_define_get(NPY_ARRAY_WRITEBACKIFCOPY, writebackifcopy)
_define_get(NPY_ARRAY_OWNDATA, owndata)
_define_get(NPY_ARRAY_ALIGNED, aligned)
_define_get(NPY_ARRAY_WRITEABLE, writeable_no_warn)
_define_get_warn(NPY_ARRAY_WRITEABLE, writeable)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE, behaved)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE|
            NPY_ARRAY_C_CONTIGUOUS, carray)

static PyObject *
arrayflags_forc_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
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
arrayflags_fnc_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
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
arrayflags_farray_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
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
arrayflags_num_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(self->flags);
}

/* relies on setflags order being write, align, uic */
static int
arrayflags_writebackifcopy_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
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
    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None, Py_None,
                              (istrue ? Py_True : Py_False));
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

static int
arrayflags_aligned_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
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
    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None,
                              (istrue ? Py_True : Py_False),
                              Py_None);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

static int
arrayflags_writeable_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
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

    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    res = PyObject_CallMethod(self->arr, "setflags", "OOO",
                              (istrue ? Py_True : Py_False),
                              Py_None, Py_None);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

static int
arrayflags_warn_on_write_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    /*
     * This code should go away in a future release, so do not mangle the
     * array_setflags function with an extra kwarg
     */
    int ret;
    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags _warn_on_write attribute");
        return -1;
    }
    ret = PyObject_IsTrue(obj);
    if (ret > 0) {
        if (!(PyArray_FLAGS((PyArrayObject*)self->arr) & NPY_ARRAY_WRITEABLE)) {
            PyErr_SetString(PyExc_ValueError,
                        "cannot set '_warn_on_write' flag when 'writable' is "
                        "False");
            return -1;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)self->arr, NPY_ARRAY_WARN_ON_WRITE);
    }
    else if (ret < 0) {
        return -1;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "cannot clear '_warn_on_write', set "
                        "writeable True to clear this private flag");
        return -1;
    }
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
    {"_writeable_no_warn",
        (getter)arrayflags_writeable_no_warn_get,
        (setter)NULL,
        NULL, NULL},
    {"_warn_on_write",
        (getter)NULL,
        (setter)arrayflags_warn_on_write_set,
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
            return arrayflags_contiguous_get(self, NULL);
        case 'F':
            return arrayflags_fortran_get(self, NULL);
        case 'W':
            return arrayflags_writeable_get(self, NULL);
        case 'B':
            return arrayflags_behaved_get(self, NULL);
        case 'O':
            return arrayflags_owndata_get(self, NULL);
        case 'A':
            return arrayflags_aligned_get(self, NULL);
        case 'X':
            return arrayflags_writebackifcopy_get(self, NULL);
        default:
            goto fail;
        }
        break;
    case 2:
        if (strncmp(key, "CA", n) == 0) {
            return arrayflags_carray_get(self, NULL);
        }
        if (strncmp(key, "FA", n) == 0) {
            return arrayflags_farray_get(self, NULL);
        }
        break;
    case 3:
        if (strncmp(key, "FNC", n) == 0) {
            return arrayflags_fnc_get(self, NULL);
        }
        break;
    case 4:
        if (strncmp(key, "FORC", n) == 0) {
            return arrayflags_forc_get(self, NULL);
        }
        break;
    case 6:
        if (strncmp(key, "CARRAY", n) == 0) {
            return arrayflags_carray_get(self, NULL);
        }
        if (strncmp(key, "FARRAY", n) == 0) {
            return arrayflags_farray_get(self, NULL);
        }
        break;
    case 7:
        if (strncmp(key,"FORTRAN",n) == 0) {
            return arrayflags_fortran_get(self, NULL);
        }
        if (strncmp(key,"BEHAVED",n) == 0) {
            return arrayflags_behaved_get(self, NULL);
        }
        if (strncmp(key,"OWNDATA",n) == 0) {
            return arrayflags_owndata_get(self, NULL);
        }
        if (strncmp(key,"ALIGNED",n) == 0) {
            return arrayflags_aligned_get(self, NULL);
        }
        break;
    case 9:
        if (strncmp(key,"WRITEABLE",n) == 0) {
            return arrayflags_writeable_get(self, NULL);
        }
        break;
    case 10:
        if (strncmp(key,"CONTIGUOUS",n) == 0) {
            return arrayflags_contiguous_get(self, NULL);
        }
        break;
    case 12:
        if (strncmp(key, "C_CONTIGUOUS", n) == 0) {
            return arrayflags_contiguous_get(self, NULL);
        }
        if (strncmp(key, "F_CONTIGUOUS", n) == 0) {
            return arrayflags_fortran_get(self, NULL);
        }
        break;
    case 15:
        if (strncmp(key, "WRITEBACKIFCOPY", n) == 0) {
            return arrayflags_writebackifcopy_get(self, NULL);
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
        return arrayflags_writeable_set(self, item, NULL);
    }
    else if (((n==7) && (strncmp(key, "ALIGNED", n) == 0)) ||
             ((n==1) && (strncmp(key, "A", n) == 0))) {
        return arrayflags_aligned_set(self, item, NULL);
    }
    else if (((n==15) && (strncmp(key, "WRITEBACKIFCOPY", n) == 0)) ||
             ((n==1) && (strncmp(key, "X", n) == 0))) {
        return arrayflags_writebackifcopy_set(self, item, NULL);
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
    const char *_warn_on_write = "";

    if (fl & NPY_ARRAY_WARN_ON_WRITE) {
        _warn_on_write = "  (with WARN_ON_WRITE=True)";
    }
    return PyUnicode_FromFormat(
                        "  %s : %s\n  %s : %s\n"
                        "  %s : %s\n  %s : %s%s\n"
                        "  %s : %s\n  %s : %s\n",
                        "C_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_C_CONTIGUOUS),
                        "F_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_F_CONTIGUOUS),
                        "OWNDATA",         _torf_(fl, NPY_ARRAY_OWNDATA),
                        "WRITEABLE",       _torf_(fl, NPY_ARRAY_WRITEABLE),
                        _warn_on_write,
                        "ALIGNED",         _torf_(fl, NPY_ARRAY_ALIGNED),
                        "WRITEBACKIFCOPY", _torf_(fl, NPY_ARRAY_WRITEBACKIFCOPY)
    );
}

static PyObject*
arrayflags_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    if (!PyObject_TypeCheck(other, &PyArrayFlags_Type)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    npy_bool eq = ((PyArrayFlagsObject*) self)->flags ==
                   ((PyArrayFlagsObject*) other)->flags;

    if (cmp_op == Py_EQ) {
        return PyBool_FromLong(eq);
    }
    else if (cmp_op == Py_NE) {
        return PyBool_FromLong(!eq);
    }
    else {
        Py_RETURN_NOTIMPLEMENTED;
    }
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
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._core.multiarray.flagsobj",
    .tp_basicsize = sizeof(PyArrayFlagsObject),
    .tp_dealloc = (destructor)arrayflags_dealloc,
    .tp_repr = (reprfunc)arrayflags_print,
    .tp_as_mapping = &arrayflags_as_mapping,
    .tp_str = (reprfunc)arrayflags_print,
    .tp_flags =Py_TPFLAGS_DEFAULT,
    .tp_richcompare = arrayflags_richcompare,
    .tp_getset = arrayflags_getsets,
    .tp_new = arrayflags_new,
};

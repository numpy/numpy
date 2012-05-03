#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "usertypes.h"

#include "common.h"
#include "buffer.h"


NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op)
{
    if (PyFloat_Check(op)) {
        return PyArray_DescrFromType(PyArray_DOUBLE);
    }
    else if (PyComplex_Check(op)) {
        return PyArray_DescrFromType(PyArray_CDOUBLE);
    }
    else if (PyInt_Check(op)) {
        /* bools are a subclass of int */
        if (PyBool_Check(op)) {
            return PyArray_DescrFromType(PyArray_BOOL);
        }
        else {
            return  PyArray_DescrFromType(PyArray_LONG);
        }
    }
    else if (PyLong_Check(op)) {
        /* if integer can fit into a longlong then return that*/
        if ((PyLong_AsLongLong(op) == -1) && PyErr_Occurred()) {
            PyErr_Clear();
            return PyArray_DescrFromType(PyArray_OBJECT);
        }
        return PyArray_DescrFromType(PyArray_LONGLONG);
    }
    return NULL;
}

static PyArray_Descr *
_use_default_type(PyObject *op)
{
    int typenum, l;
    PyObject *type;

    typenum = -1;
    l = 0;
    type = (PyObject *)Py_TYPE(op);
    while (l < PyArray_NUMUSERTYPES) {
        if (type == (PyObject *)(userdescrs[l]->typeobj)) {
            typenum = l + PyArray_USERDEF;
            break;
        }
        l++;
    }
    if (typenum == -1) {
        typenum = PyArray_OBJECT;
    }
    return PyArray_DescrFromType(typenum);
}


/*
 * op is an object to be converted to an ndarray.
 *
 * minitype is the minimum type-descriptor needed.
 *
 * max is the maximum number of dimensions -- used for recursive call
 * to avoid infinite recursion...
 */
NPY_NO_EXPORT PyArray_Descr *
_array_find_type(PyObject *op, PyArray_Descr *minitype, int max)
{
    int l;
    PyObject *ip;
    PyArray_Descr *chktype = NULL;
    PyArray_Descr *outtype;
#if PY_VERSION_HEX >= 0x02060000
    Py_buffer buffer_view;
#endif

    /*
     * These need to come first because if op already carries
     * a descr structure, then we want it to be the result if minitype
     * is NULL.
     */
    if (PyArray_Check(op)) {
        chktype = PyArray_DESCR(op);
        Py_INCREF(chktype);
        if (minitype == NULL) {
            return chktype;
        }
        Py_INCREF(minitype);
        goto finish;
    }

    if (PyArray_IsScalar(op, Generic)) {
        chktype = PyArray_DescrFromScalar(op);
        if (minitype == NULL) {
            return chktype;
        }
        Py_INCREF(minitype);
        goto finish;
    }

    if (minitype == NULL) {
        minitype = PyArray_DescrFromType(PyArray_BOOL);
    }
    else {
        Py_INCREF(minitype);
    }
    if (max < 0) {
        goto deflt;
    }
    chktype = _array_find_python_scalar_type(op);
    if (chktype) {
        goto finish;
    }

    if (PyBytes_Check(op)) {
        chktype = PyArray_DescrNewFromType(PyArray_STRING);
        chktype->elsize = PyString_GET_SIZE(op);
        goto finish;
    }

    if (PyUnicode_Check(op)) {
        chktype = PyArray_DescrNewFromType(PyArray_UNICODE);
        chktype->elsize = PyUnicode_GET_DATA_SIZE(op);
#ifndef Py_UNICODE_WIDE
        chktype->elsize <<= 1;
#endif
        goto finish;
    }

#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 buffer interface */
    memset(&buffer_view, 0, sizeof(Py_buffer));
    if (PyObject_GetBuffer(op, &buffer_view, PyBUF_FORMAT|PyBUF_STRIDES) == 0 ||
        PyObject_GetBuffer(op, &buffer_view, PyBUF_FORMAT) == 0) {

        PyErr_Clear();
        chktype = _descriptor_from_pep3118_format(buffer_view.format);
        PyBuffer_Release(&buffer_view);
        if (chktype) {
            goto finish;
        }
    }
    else if (PyObject_GetBuffer(op, &buffer_view, PyBUF_STRIDES) == 0 ||
             PyObject_GetBuffer(op, &buffer_view, PyBUF_SIMPLE) == 0) {

        PyErr_Clear();
        chktype = PyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = buffer_view.itemsize;
        PyBuffer_Release(&buffer_view);
        goto finish;
    }
    else {
        PyErr_Clear();
    }
#endif

    if ((ip=PyObject_GetAttrString(op, "__array_interface__"))!=NULL) {
        if (PyDict_Check(ip)) {
            PyObject *new;
            new = PyDict_GetItemString(ip, "typestr");
            if (new && PyString_Check(new)) {
                chktype =_array_typedescr_fromstr(PyString_AS_STRING(new));
            }
        }
        Py_DECREF(ip);
        if (chktype) {
            goto finish;
        }
    }
    else {
        PyErr_Clear();
    }
    if ((ip=PyObject_GetAttrString(op, "__array_struct__")) != NULL) {
        PyArrayInterface *inter;
        char buf[40];

        if (NpyCapsule_Check(ip)) {
            inter = (PyArrayInterface *)NpyCapsule_AsVoidPtr(ip);
            if (inter->two == 2) {
                PyOS_snprintf(buf, sizeof(buf),
                        "|%c%d", inter->typekind, inter->itemsize);
                chktype = _array_typedescr_fromstr(buf);
            }
        }
        Py_DECREF(ip);
        if (chktype) {
            goto finish;
        }
    }
    else {
        PyErr_Clear();
    }

#if !defined(NPY_PY3K)
    if (PyBuffer_Check(op)) {
        chktype = PyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = Py_TYPE(op)->tp_as_sequence->sq_length(op);
        PyErr_Clear();
        goto finish;
    }
#endif

    if (PyObject_HasAttrString(op, "__array__")) {
        ip = PyObject_CallMethod(op, "__array__", NULL);
        if(ip && PyArray_Check(ip)) {
            chktype = PyArray_DESCR(ip);
            Py_INCREF(chktype);
            Py_DECREF(ip);
            goto finish;
        }
        Py_XDECREF(ip);
        if (PyErr_Occurred()) PyErr_Clear();
    }

#if defined(NPY_PY3K)
    /* FIXME: XXX -- what is the correct thing to do here? */
#else
    if (PyInstance_Check(op)) {
        goto deflt;
    }
#endif
    if (PySequence_Check(op)) {
        l = PyObject_Length(op);
        if (l < 0 && PyErr_Occurred()) {
            PyErr_Clear();
            goto deflt;
        }
        if (l == 0 && minitype->type_num == PyArray_BOOL) {
            Py_DECREF(minitype);
            minitype = PyArray_DescrFromType(PyArray_DEFAULT);
            if (minitype == NULL) {
                return NULL;
            }
        }
        while (--l >= 0) {
            PyArray_Descr *newtype;
            ip = PySequence_GetItem(op, l);
            if (ip==NULL) {
                PyErr_Clear();
                goto deflt;
            }
            chktype = _array_find_type(ip, minitype, max-1);
            if (chktype == NULL) {
                Py_DECREF(minitype);
                return NULL;
            }
            newtype = PyArray_PromoteTypes(chktype, minitype);
            Py_DECREF(minitype);
            minitype = newtype;
            Py_DECREF(chktype);
            Py_DECREF(ip);
        }
        chktype = minitype;
        Py_INCREF(minitype);
        goto finish;
    }


 deflt:
    chktype = _use_default_type(op);

 finish:
    outtype = PyArray_PromoteTypes(chktype, minitype);
    Py_DECREF(chktype);
    Py_DECREF(minitype);
    if (outtype == NULL) {
        return NULL;
    }
    /*
     * VOID Arrays should not occur by "default"
     * unless input was already a VOID
     */
    if (outtype->type_num == PyArray_VOID &&
        minitype->type_num != PyArray_VOID) {
        Py_DECREF(outtype);
        return PyArray_DescrFromType(PyArray_OBJECT);
    }
    return outtype;
}

/* new reference */
NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *c_str)
{
    PyArray_Descr *descr = NULL;
    PyObject *stringobj = PyString_FromString(c_str);

    if (stringobj == NULL) {
        return NULL;
    }
    if (PyArray_DescrConverter(stringobj, &descr) != NPY_SUCCEED) {
        Py_DECREF(stringobj);
        return NULL;
    }
    Py_DECREF(stringobj);
    return descr;
}

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, intp i)
{
    intp dim0;

    if (mp->nd == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = mp->dimensions[0];
    if (i < 0) {
        i += dim0;
    }
    if (i == 0 && dim0 > 0) {
        return mp->data;
    }
    if (i > 0 && i < dim0) {
        return mp->data+i*mp->strides[0];
    }
    PyErr_SetString(PyExc_IndexError,"index out of bounds");
    return NULL;
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(ret->descr)) {
        PyObject *zero = PyInt_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        intp n = PyArray_NBYTES(ret);
        memset(ret->data, 0, n);
    }
    return 0;
}

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap)
{
    int i, alignment, aligned = 1;
    intp ptr;

    /* The special casing for STRING and VOID types was removed
     * in accordance with http://projects.scipy.org/numpy/ticket/1227
     * It used to be that IsAligned always returned True for these
     * types, which is indeed the case when they are created using
     * PyArray_DescrConverter(), but not necessarily when using
     * PyArray_DescrAlignConverter(). */

    alignment = ap->descr->alignment;
    if (alignment == 1) {
        return 1;
    }
    ptr = (intp) ap->data;
    aligned = (ptr % alignment) == 0;
    for (i = 0; i < ap->nd; i++) {
        aligned &= ((ap->strides[i] % alignment) == 0);
    }
    return aligned != 0;
}

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base=ap->base;
    void *dummy;
    Py_ssize_t n;

    /* If we own our own data, then no-problem */
    if ((base == NULL) || (ap->flags & OWNDATA)) {
        return TRUE;
    }
    /*
     * Get to the final base object
     * If it is a writeable array, then return TRUE
     * If we can find an array object
     * or a writeable buffer object as the final base object
     * or a string object (for pickling support memory savings).
     * - this last could be removed if a proper pickleable
     * buffer was added to Python.
     */

    while(PyArray_Check(base)) {
        if (PyArray_CHKFLAGS(base, OWNDATA)) {
            return (Bool) (PyArray_ISWRITEABLE(base));
        }
        base = PyArray_BASE(base);
    }

    /*
     * here so pickle support works seamlessly
     * and unpickled array can be set and reset writeable
     * -- could be abused --
     */
    if (PyString_Check(base)) {
        return TRUE;
    }
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        return FALSE;
    }
    return TRUE;
}

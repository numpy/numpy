#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_3kcompat.h"

#include "usertypes.h"

#include "common.h"

/*
 * new reference
 * doesn't alter refcount of chktype or mintype ---
 * unless one of them is returned
 */
NPY_NO_EXPORT PyArray_Descr *
_array_small_type(PyArray_Descr *chktype, PyArray_Descr* mintype)
{
    PyArray_Descr *outtype;
    int outtype_num, save_num;

    if (PyArray_EquivTypes(chktype, mintype)) {
        Py_INCREF(mintype);
        return mintype;
    }


    if (chktype->type_num > mintype->type_num) {
        outtype_num = chktype->type_num;
    }
    else {
        if (PyDataType_ISOBJECT(chktype) &&
            PyDataType_ISSTRING(mintype)) {
            return PyArray_DescrFromType(NPY_OBJECT);
        }
        else {
            outtype_num = mintype->type_num;
        }
    }

    save_num = outtype_num;
    while (outtype_num < PyArray_NTYPES &&
          !(PyArray_CanCastSafely(chktype->type_num, outtype_num)
            && PyArray_CanCastSafely(mintype->type_num, outtype_num))) {
        outtype_num++;
    }
    if (outtype_num == PyArray_NTYPES) {
        outtype = PyArray_DescrFromType(save_num);
    }
    else {
        outtype = PyArray_DescrFromType(outtype_num);
    }
    if (PyTypeNum_ISEXTENDED(outtype->type_num)) {
        int testsize = outtype->elsize;
        int chksize, minsize;
        chksize = chktype->elsize;
        minsize = mintype->elsize;
        /*
         * Handle string->unicode case separately
         * because string itemsize is 4* as large
         */
        if (outtype->type_num == PyArray_UNICODE &&
            mintype->type_num == PyArray_STRING) {
            testsize = MAX(chksize, 4*minsize);
        }
        else if (chktype->type_num == PyArray_STRING &&
                 mintype->type_num == PyArray_UNICODE) {
            testsize = MAX(chksize*4, minsize);
        }
        else {
            testsize = MAX(chksize, minsize);
        }
        if (testsize != outtype->elsize) {
            PyArray_DESCR_REPLACE(outtype);
            outtype->elsize = testsize;
            Py_XDECREF(outtype->fields);
            outtype->fields = NULL;
            Py_XDECREF(outtype->names);
            outtype->names = NULL;
        }
    }
    return outtype;
}

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
        if (PyCObject_Check(ip)) {
            inter=(PyArrayInterface *)PyCObject_AsVoidPtr(ip);
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

    if (PyString_Check(op)) {
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

#if defined(NPY_PY3K)
    if (PyMemoryView_Check(op)) {
#else
    if (PyBuffer_Check(op)) {
#endif
        chktype = PyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = Py_TYPE(op)->tp_as_sequence->sq_length(op);
        PyErr_Clear();
        goto finish;
    }

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
#warning XXX -- what is the correct thing to do here?
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
        }
        while (--l >= 0) {
            PyArray_Descr *newtype;
            ip = PySequence_GetItem(op, l);
            if (ip==NULL) {
                PyErr_Clear();
                goto deflt;
            }
            chktype = _array_find_type(ip, minitype, max-1);
            newtype = _array_small_type(chktype, minitype);
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
    outtype = _array_small_type(chktype, minitype);
    Py_DECREF(chktype);
    Py_DECREF(minitype);
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
_array_typedescr_fromstr(char *str)
{
    PyArray_Descr *descr;
    int type_num;
    char typechar;
    int size;
    char msg[] = "unsupported typestring";
    int swap;
    char swapchar;

    swapchar = str[0];
    str += 1;

    typechar = str[0];
    size = atoi(str + 1);
    switch (typechar) {
    case 'b':
        if (size == sizeof(Bool)) {
            type_num = PyArray_BOOL;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'u':
        if (size == sizeof(uintp)) {
            type_num = PyArray_UINTP;
        }
        else if (size == sizeof(char)) {
            type_num = PyArray_UBYTE;
        }
        else if (size == sizeof(short)) {
            type_num = PyArray_USHORT;
        }
        else if (size == sizeof(ulong)) {
            type_num = PyArray_ULONG;
        }
        else if (size == sizeof(int)) {
            type_num = PyArray_UINT;
        }
        else if (size == sizeof(ulonglong)) {
            type_num = PyArray_ULONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'i':
        if (size == sizeof(intp)) {
            type_num = PyArray_INTP;
        }
        else if (size == sizeof(char)) {
            type_num = PyArray_BYTE;
        }
        else if (size == sizeof(short)) {
            type_num = PyArray_SHORT;
        }
        else if (size == sizeof(long)) {
            type_num = PyArray_LONG;
        }
        else if (size == sizeof(int)) {
            type_num = PyArray_INT;
        }
        else if (size == sizeof(longlong)) {
            type_num = PyArray_LONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'f':
        if (size == sizeof(float)) {
            type_num = PyArray_FLOAT;
        }
        else if (size == sizeof(double)) {
            type_num = PyArray_DOUBLE;
        }
        else if (size == sizeof(longdouble)) {
            type_num = PyArray_LONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'c':
        if (size == sizeof(float)*2) {
            type_num = PyArray_CFLOAT;
        }
        else if (size == sizeof(double)*2) {
            type_num = PyArray_CDOUBLE;
        }
        else if (size == sizeof(longdouble)*2) {
            type_num = PyArray_CLONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'O':
        if (size == sizeof(PyObject *)) {
            type_num = PyArray_OBJECT;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case PyArray_STRINGLTR:
        type_num = PyArray_STRING;
        break;
    case PyArray_UNICODELTR:
        type_num = PyArray_UNICODE;
        size <<= 2;
        break;
    case 'V':
        type_num = PyArray_VOID;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    swap = !PyArray_ISNBO(swapchar);
    if (descr->elsize == 0 || swap) {
        /* Need to make a new PyArray_Descr */
        PyArray_DESCR_REPLACE(descr);
        if (descr==NULL) {
            return NULL;
        }
        if (descr->elsize == 0) {
            descr->elsize = size;
        }
        if (swap) {
            descr->byteorder = swapchar;
        }
    }
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
    if PyString_Check(base) {
        return TRUE;
    }
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        return FALSE;
    }
    return TRUE;
}

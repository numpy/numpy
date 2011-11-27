#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API
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

#if !defined(NPY_PY3K)
static PyArray_Descr *
_use_default_type(PyObject *op)
{
    int typenum, l;
    PyObject *type;

    typenum = -1;
    l = 0;
    type = (PyObject *)Py_TYPE(op);
    while (l < NPY_NUMUSERTYPES) {
        if (type == (PyObject *)(userdescrs[l]->typeobj)) {
            typenum = l + NPY_USERDEF;
            break;
        }
        l++;
    }
    if (typenum == -1) {
        typenum = NPY_OBJECT;
    }
    return PyArray_DescrFromType(typenum);
}
#endif

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_contains_na' gets set to 1 if an np.NA object is encountered.
 * The NA does not affect the dtype produced, so if this is set to 1
 * and the result is for an array without NA support, the dtype should
 * be switched to NPY_OBJECT. When adding multi-NA support, this should
 * also signal whether just regular NAs or NAs with payloads were seen.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 * This is reset to NULL on failure.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, int *out_contains_na,
                        PyArray_Descr **out_dtype)
{
    int i, size;
    PyArray_Descr *dtype = NULL;
    PyObject *ip;
#if PY_VERSION_HEX >= 0x02060000
    Py_buffer buffer_view;
#endif

    /* Check if it's an ndarray */
    if (PyArray_Check(obj)) {
        /* Check for any NAs in the array */
        int containsna = PyArray_ContainsNA((PyArrayObject *)obj, NULL, NULL);
        if (containsna == -1) {
            goto fail;
        }
        else if (containsna) {
            *out_contains_na = 1;
        }
        dtype = PyArray_DESCR((PyArrayObject *)obj);
        Py_INCREF(dtype);
        goto promote_types;
    }

    /* Check if it's a NumPy scalar */
    if (PyArray_IsScalar(obj, Generic)) {
        dtype = PyArray_DescrFromScalar(obj);
        if (dtype == NULL) {
            goto fail;
        }
        goto promote_types;
    }

    /* Check if it's a Python scalar */
    dtype = _array_find_python_scalar_type(obj);
    if (dtype != NULL) {
        goto promote_types;
    }

    /* Check if it's an NA */
    if (NpyNA_Check(obj)) {
        *out_contains_na = 1;
        return 0;
    }

    /* Check if it's an ASCII string */
    if (PyBytes_Check(obj)) {
        int itemsize = PyString_GET_SIZE(obj);

        /* If it's already a big enough string, don't bother type promoting */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_STRING &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        dtype = PyArray_DescrNewFromType(NPY_STRING);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = itemsize;
        goto promote_types;
    }

    /* Check if it's a Unicode string */
    if (PyUnicode_Check(obj)) {
        int itemsize = PyUnicode_GET_DATA_SIZE(obj);
#ifndef Py_UNICODE_WIDE
        itemsize <<= 1;
#endif

        /*
         * If it's already a big enough unicode object,
         * don't bother type promoting
         */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_UNICODE &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        dtype = PyArray_DescrNewFromType(NPY_UNICODE);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = itemsize;
        goto promote_types;
    }

#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 buffer interface */
    memset(&buffer_view, 0, sizeof(Py_buffer));
    if (PyObject_GetBuffer(obj, &buffer_view, PyBUF_FORMAT|PyBUF_STRIDES) == 0 ||
        PyObject_GetBuffer(obj, &buffer_view, PyBUF_FORMAT) == 0) {

        PyErr_Clear();
        dtype = _descriptor_from_pep3118_format(buffer_view.format);
        PyBuffer_Release(&buffer_view);
        if (dtype) {
            goto promote_types;
        }
    }
    else if (PyObject_GetBuffer(obj, &buffer_view, PyBUF_STRIDES) == 0 ||
             PyObject_GetBuffer(obj, &buffer_view, PyBUF_SIMPLE) == 0) {

        PyErr_Clear();
        dtype = PyArray_DescrNewFromType(NPY_VOID);
        dtype->elsize = buffer_view.itemsize;
        PyBuffer_Release(&buffer_view);
        goto promote_types;
    }
    else {
        PyErr_Clear();
    }
#endif

    /* The array interface */
    ip = PyObject_GetAttrString(obj, "__array_interface__");
    if (ip != NULL) {
        if (PyDict_Check(ip)) {
            PyObject *typestr;
            typestr = PyDict_GetItemString(ip, "typestr");
            if (typestr && PyString_Check(typestr)) {
                dtype =_array_typedescr_fromstr(PyString_AS_STRING(typestr));
                Py_DECREF(ip);
                if (dtype == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }
    else {
        PyErr_Clear();
    }

    /* The array struct interface */
    ip = PyObject_GetAttrString(obj, "__array_struct__");
    if (ip != NULL) {
        PyArrayInterface *inter;
        char buf[40];

        if (NpyCapsule_Check(ip)) {
            inter = (PyArrayInterface *)NpyCapsule_AsVoidPtr(ip);
            if (inter->two == 2) {
                PyOS_snprintf(buf, sizeof(buf),
                        "|%c%d", inter->typekind, inter->itemsize);
                dtype = _array_typedescr_fromstr(buf);
                Py_DECREF(ip);
                if (dtype == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }
    else {
        PyErr_Clear();
    }

    /* The old buffer interface */
#if !defined(NPY_PY3K)
    if (PyBuffer_Check(obj)) {
        dtype = PyArray_DescrNewFromType(NPY_VOID);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = Py_TYPE(obj)->tp_as_sequence->sq_length(obj);
        PyErr_Clear();
        goto promote_types;
    }
#endif

    /* The __array__ attribute */
    if (PyObject_HasAttrString(obj, "__array__")) {
        ip = PyObject_CallMethod(obj, "__array__", NULL);
        if(ip && PyArray_Check(ip)) {
            dtype = PyArray_DESCR((PyArrayObject *)ip);
            Py_INCREF(dtype);
            Py_DECREF(ip);
            goto promote_types;
        }
        Py_XDECREF(ip);
        if (PyErr_Occurred()) {
            goto fail;
        }
    }

    /* Not exactly sure what this is about... */
#if !defined(NPY_PY3K)
    if (PyInstance_Check(obj)) {
        dtype = _use_default_type(obj);
        if (dtype == NULL) {
            goto fail;
        }
        else {
            goto promote_types;
        }
    }
#endif

    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, the output dtype should be OBJECT
     */
    if (maxdims == 0 || !PySequence_Check(obj)) {
        if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
            Py_XDECREF(*out_dtype);
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
        }
        return 0;
    }

    /* Recursive case */
    size = PySequence_Size(obj);
    if (size < 0) {
        goto fail;
    }
    /* Recursive call for each sequence item */
    for (i = 0; i < size; ++i) {
        ip = PySequence_GetItem(obj, i);
        if (ip==NULL) {
            goto fail;
        }
        if (PyArray_DTypeFromObject(ip, maxdims - 1,
                            out_contains_na, out_dtype) < 0) {
            Py_DECREF(ip);
            goto fail;
        }
        Py_DECREF(ip);
    }

    return 0;


promote_types:
    /* Set 'out_dtype' if it's NULL */
    if (*out_dtype == NULL) {
        *out_dtype = dtype;
        return 0;
    }
    /* Do type promotion with 'out_dtype' */
    else {
        PyArray_Descr *res_dtype = PyArray_PromoteTypes(dtype, *out_dtype);
        Py_DECREF(dtype);
        if (res_dtype == NULL) {
            return -1;
        }
        Py_DECREF(*out_dtype);
        *out_dtype = res_dtype;
        return 0;
    }

fail:
    Py_XDECREF(*out_dtype);
    *out_dtype = NULL;
    return -1;
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
            type_num = NPY_BOOL;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'u':
        if (size == sizeof(uintp)) {
            type_num = NPY_UINTP;
        }
        else if (size == sizeof(char)) {
            type_num = NPY_UBYTE;
        }
        else if (size == sizeof(short)) {
            type_num = NPY_USHORT;
        }
        else if (size == sizeof(ulong)) {
            type_num = NPY_ULONG;
        }
        else if (size == sizeof(int)) {
            type_num = NPY_UINT;
        }
        else if (size == sizeof(ulonglong)) {
            type_num = NPY_ULONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'i':
        if (size == sizeof(intp)) {
            type_num = NPY_INTP;
        }
        else if (size == sizeof(char)) {
            type_num = NPY_BYTE;
        }
        else if (size == sizeof(short)) {
            type_num = NPY_SHORT;
        }
        else if (size == sizeof(long)) {
            type_num = NPY_LONG;
        }
        else if (size == sizeof(int)) {
            type_num = NPY_INT;
        }
        else if (size == sizeof(longlong)) {
            type_num = NPY_LONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'f':
        if (size == sizeof(float)) {
            type_num = NPY_FLOAT;
        }
        else if (size == sizeof(double)) {
            type_num = NPY_DOUBLE;
        }
        else if (size == sizeof(longdouble)) {
            type_num = NPY_LONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'c':
        if (size == sizeof(float)*2) {
            type_num = NPY_CFLOAT;
        }
        else if (size == sizeof(double)*2) {
            type_num = NPY_CDOUBLE;
        }
        else if (size == sizeof(longdouble)*2) {
            type_num = NPY_CLONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'O':
        if (size == sizeof(PyObject *)) {
            type_num = NPY_OBJECT;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case NPY_STRINGLTR:
        type_num = NPY_STRING;
        break;
    case NPY_UNICODELTR:
        type_num = NPY_UNICODE;
        size <<= 2;
        break;
    case 'V':
        type_num = NPY_VOID;
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

    if (PyArray_NDIM(mp) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = PyArray_DIMS(mp)[0];
    if (i < 0) {
        i += dim0;
    }
    if (i == 0 && dim0 > 0) {
        return PyArray_DATA(mp);
    }
    if (i > 0 && i < dim0) {
        return PyArray_DATA(mp)+i*PyArray_STRIDES(mp)[0];
    }
    PyErr_SetString(PyExc_IndexError,"index out of bounds");
    return NULL;
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
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
        memset(PyArray_DATA(ret), 0, n);
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

    alignment = PyArray_DESCR(ap)->alignment;
    if (alignment == 1) {
        return 1;
    }
    ptr = (intp) PyArray_DATA(ap);
    aligned = (ptr % alignment) == 0;
    for (i = 0; i < PyArray_NDIM(ap); i++) {
        aligned &= ((PyArray_STRIDES(ap)[i] % alignment) == 0);
    }
    return aligned != 0;
}

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base=PyArray_BASE(ap);
    void *dummy;
    Py_ssize_t n;

    /* If we own our own data, then no-problem */
    if ((base == NULL) || (PyArray_FLAGS(ap) & NPY_ARRAY_OWNDATA)) {
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
     *
     * MW: I think it would better to disallow switching from READONLY
     *     to WRITEABLE like this...
     */

    while(PyArray_Check(base)) {
        if (PyArray_CHKFLAGS((PyArrayObject *)base, NPY_ARRAY_OWNDATA)) {
            return (npy_bool) (PyArray_ISWRITEABLE((PyArrayObject *)base));
        }
        base = PyArray_BASE((PyArrayObject *)base);
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

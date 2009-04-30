/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young Univeristy


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "arrayobject.h"
#include "arraydescr.h"
#include "arrayiterators.h"
#include "arraymapping.h"
#ifndef Py_UNICODE_WIDE
#include "ucsnarrow.h"
#endif



/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = PyArray_PRIORITY;

    if (PyArray_CheckExact(obj))
        return priority;

    ret = PyObject_GetAttrString(obj, "__array_priority__");
    if (ret != NULL) {
        priority = PyFloat_AsDouble(ret);
    }
    if (PyErr_Occurred()) {
        PyErr_Clear();
        priority = default_;
    }
    Py_XDECREF(ret);
    return priority;
}

static int
_check_object_rec(PyArray_Descr *descr)
{
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

/* Backward compatibility only */
/* In both Zero and One

***You must free the memory once you are done with it
using PyDataMem_FREE(ptr) or you create a memory leak***

If arr is an Object array you are getting a
BORROWED reference to Zero or One.
Do not DECREF.
Please INCREF if you will be hanging on to it.

The memory for the ptr still must be freed in any case;
*/


/*NUMPY_API
  Get pointer to zero of correct type for array.
*/
NPY_NO_EXPORT char *
PyArray_Zero(PyArrayObject *arr)
{
    char *zeroval;
    int ret, storeflags;
    PyObject *obj;

    if (_check_object_rec(arr->descr) < 0) {
        return NULL;
    }
    zeroval = PyDataMem_NEW(arr->descr->elsize);
    if (zeroval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    obj=PyInt_FromLong((long) 0);
    if (PyArray_ISOBJECT(arr)) {
        memcpy(zeroval, &obj, sizeof(PyObject *));
        Py_DECREF(obj);
        return zeroval;
    }
    storeflags = arr->flags;
    arr->flags |= BEHAVED;
    ret = arr->descr->f->setitem(obj, zeroval, arr);
    arr->flags = storeflags;
    Py_DECREF(obj);
    if (ret < 0) {
        PyDataMem_FREE(zeroval);
        return NULL;
    }
    return zeroval;
}

/*NUMPY_API
  Get pointer to one of correct type for array
*/
NPY_NO_EXPORT char *
PyArray_One(PyArrayObject *arr)
{
    char *oneval;
    int ret, storeflags;
    PyObject *obj;

    if (_check_object_rec(arr->descr) < 0) {
        return NULL;
    }
    oneval = PyDataMem_NEW(arr->descr->elsize);
    if (oneval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    obj = PyInt_FromLong((long) 1);
    if (PyArray_ISOBJECT(arr)) {
        memcpy(oneval, &obj, sizeof(PyObject *));
        Py_DECREF(obj);
        return oneval;
    }

    storeflags = arr->flags;
    arr->flags |= BEHAVED;
    ret = arr->descr->f->setitem(obj, oneval, arr);
    arr->flags = storeflags;
    Py_DECREF(obj);
    if (ret < 0) {
        PyDataMem_FREE(oneval);
        return NULL;
    }
    return oneval;
}

/* End deprecated */


NPY_NO_EXPORT PyObject *PyArray_New(PyTypeObject *, int nd, intp *,
                             int, intp *, void *, int, int, PyObject *);


/* Incref all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject **temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == PyArray_OBJECT) {
        temp = (PyObject **)data;
        Py_XINCREF(*temp);
    }
    else if (PyDescr_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            PyArray_Item_INCREF(data + offset, new);
        }
    }
    return;
}

/* XDECREF all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject **temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == PyArray_OBJECT) {
        temp = (PyObject **)data;
        Py_XDECREF(*temp);
    }
    else if PyDescr_HASFIELDS(descr) {
            PyObject *key, *value, *title = NULL;
            PyArray_Descr *new;
            int offset;
            Py_ssize_t pos = 0;

            while (PyDict_Next(descr->fields, &pos, &key, &value)) {
		if NPY_TITLE_KEY(key, value) {
                    continue;
                }
		if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                      &title)) {
                    return;
                }
                PyArray_Item_XDECREF(data + offset, new);
            }
        }
    return;
}

/* C-API functions */

/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    intp i, n;
    PyObject **data, **temp;
    PyArrayIterObject *it;

    if (!PyDataType_REFCHK(mp->descr)) {
        return 0;
    }
    if (mp->descr->type_num != PyArray_OBJECT) {
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, mp->descr);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)mp->data;
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                temp = data;
                Py_XINCREF(*temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            temp = (PyObject **)it->dataptr;
            Py_XINCREF(*temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    intp i, n;
    PyObject **data;
    PyObject **temp;
    PyArrayIterObject *it;

    if (!PyDataType_REFCHK(mp->descr)) {
        return 0;
    }
    if (mp->descr->type_num != PyArray_OBJECT) {
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_XDECREF(it->dataptr, mp->descr);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)mp->data;
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                temp = data;
                Py_XDECREF(*temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            temp = (PyObject **)it->dataptr;
            Py_XDECREF(*temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

static void
_strided_byte_copy(char *dst, intp outstrides, char *src, intp instrides,
                   intp N, int elsize)
{
    intp i, j;
    char *tout = dst;
    char *tin = src;

#define _FAST_MOVE(_type_)                              \
    for(i=0; i<N; i++) {                               \
        ((_type_ *)tout)[0] = ((_type_ *)tin)[0];       \
        tin += instrides;                               \
        tout += outstrides;                             \
    }                                                   \
    return

    switch(elsize) {
    case 8:
        _FAST_MOVE(Int64);
    case 4:
        _FAST_MOVE(Int32);
    case 1:
        _FAST_MOVE(Int8);
    case 2:
        _FAST_MOVE(Int16);
    case 16:
        for (i = 0; i < N; i++) {
            ((Int64 *)tout)[0] = ((Int64 *)tin)[0];
            ((Int64 *)tout)[1] = ((Int64 *)tin)[1];
            tin += instrides;
            tout += outstrides;
        }
        return;
    default:
        for(i = 0; i < N; i++) {
            for(j=0; j<elsize; j++) {
                *tout++ = *tin++;
            }
            tin = tin + instrides - elsize;
            tout = tout + outstrides - elsize;
        }
    }
#undef _FAST_MOVE

}


static void
_unaligned_strided_byte_move(char *dst, intp outstrides, char *src,
                             intp instrides, intp N, int elsize)
{
    intp i;
    char *tout = dst;
    char *tin = src;


#define _MOVE_N_SIZE(size)                      \
    for(i=0; i<N; i++) {                       \
        memmove(tout, tin, size);               \
        tin += instrides;                       \
        tout += outstrides;                     \
    }                                           \
    return

    switch(elsize) {
    case 8:
        _MOVE_N_SIZE(8);
    case 4:
        _MOVE_N_SIZE(4);
    case 1:
        _MOVE_N_SIZE(1);
    case 2:
        _MOVE_N_SIZE(2);
    case 16:
        _MOVE_N_SIZE(16);
    default:
        _MOVE_N_SIZE(elsize);
    }
#undef _MOVE_N_SIZE

}

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, intp outstrides, char *src,
                             intp instrides, intp N, int elsize)
{
    intp i;
    char *tout = dst;
    char *tin = src;

#define _COPY_N_SIZE(size)                      \
    for(i=0; i<N; i++) {                       \
        memcpy(tout, tin, size);                \
        tin += instrides;                       \
        tout += outstrides;                     \
    }                                           \
    return

    switch(elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE

}

NPY_NO_EXPORT void
_strided_byte_swap(void *p, intp stride, intp n, int size)
{
    char *a, *b, c = 0;
    int j, m;

    switch(size) {
    case 1: /* no byteswap necessary */
        break;
    case 4:
        for (a = (char*)p; n > 0; n--, a += stride - 1) {
            b = a + 3;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 8:
        for (a = (char*)p; n > 0; n--, a += stride - 3) {
            b = a + 7;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 2:
        for (a = (char*)p; n > 0; n--, a += stride) {
            b = a + 1;
            c = *a; *a = *b; *b = c;
        }
        break;
    default:
        m = size/2;
        for (a = (char *)p; n > 0; n--, a += stride - m) {
            b = a + (size - 1);
            for (j = 0; j < m; j++) {
                c=*a; *a++ = *b; *b-- = c;
            }
        }
        break;
    }
}

NPY_NO_EXPORT void
byte_swap_vector(void *p, intp n, int size)
{
    _strided_byte_swap(p, (intp) size, n, size);
    return;
}

/* If numitems > 1, then dst must be contiguous */
NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, intp numitems,
              intp srcstrides, int swap)
{
    intp i;
    char *s1 = (char *)src;
    char *d1 = (char *)dst;


    if ((numitems == 1) || (itemsize == srcstrides)) {
        memcpy(d1, s1, itemsize*numitems);
    }
    else {
        for (i = 0; i < numitems; i++) {
            memcpy(d1, s1, itemsize);
            d1 += itemsize;
            s1 += srcstrides;
        }
    }

    if (swap) {
        byte_swap_vector(d1, numitems, itemsize);
    }
}


NPY_NO_EXPORT PyArray_Descr **userdescrs=NULL;


/* Helper functions */

/*NUMPY_API*/
NPY_NO_EXPORT intp
PyArray_PyIntAsIntp(PyObject *o)
{
    longlong long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    PyObject *arr;
    PyArray_Descr *descr;
    intp ret;

    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (longlong) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (longlong) PyLong_AsLongLong(o);
        goto finish;
    }

#if SIZEOF_INTP == SIZEOF_LONG
    descr = &LONG_Descr;
#elif SIZEOF_INTP == SIZEOF_INT
    descr = &INT_Descr;
#else
    descr = &LONGLONG_Descr;
#endif
    arr = NULL;

    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Py_INCREF(descr);
        arr = PyArray_CastToType((PyArrayObject *)o, descr, 0);
    }
    else if (PyArray_IsScalar(o, Integer)) {
        Py_INCREF(descr);
        arr = PyArray_FromScalar(o, descr);
    }
    if (arr != NULL) {
        ret = *((intp *)PyArray_DATA(arr));
        Py_DECREF(arr);
        return ret;
    }

#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        if (value == NULL) {
            return -1;
        }
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
    if (o->ob_type->tp_as_number != NULL &&                 \
        o->ob_type->tp_as_number->nb_long != NULL) {
        obj = o->ob_type->tp_as_number->nb_long(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else if (o->ob_type->tp_as_number != NULL &&            \
             o->ob_type->tp_as_number->nb_int != NULL) {
        obj = o->ob_type->tp_as_number->nb_int(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (SIZEOF_LONGLONG > SIZEOF_INTP)
    if ((long_value < MIN_INTP) || (long_value > MAX_INTP)) {
        PyErr_SetString(PyExc_ValueError,
                        "integer won't fit into a C intp");
        return -1;
    }
#endif
    return (intp) long_value;
}


static PyObject *array_int(PyArrayObject *v);

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o)
{
    long long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    PyObject *arr;
    PyArray_Descr *descr;
    int ret;


    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (long) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (long) PyLong_AsLong(o);
        goto finish;
    }

    descr = &INT_Descr;
    arr = NULL;
    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Py_INCREF(descr);
        arr = PyArray_CastToType((PyArrayObject *)o, descr, 0);
    }
    if (PyArray_IsScalar(o, Integer)) {
        Py_INCREF(descr);
        arr = PyArray_FromScalar(o, descr);
    }
    if (arr != NULL) {
        ret = *((int *)PyArray_DATA(arr));
        Py_DECREF(arr);
        return ret;
    }
#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
    if (o->ob_type->tp_as_number != NULL &&         \
        o->ob_type->tp_as_number->nb_int != NULL) {
        obj = o->ob_type->tp_as_number->nb_int(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
    else if (o->ob_type->tp_as_number != NULL &&                    \
             o->ob_type->tp_as_number->nb_long != NULL) {
        obj = o->ob_type->tp_as_number->nb_long(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (SIZEOF_LONG > SIZEOF_INT)
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    return (int) long_value;
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

/*NUMPY_API
  Compute the size of an array (in number of items)
*/
NPY_NO_EXPORT intp
PyArray_Size(PyObject *op)
{
    if (PyArray_Check(op)) {
        return PyArray_SIZE((PyArrayObject *)op);
    }
    else {
        return 0;
    }
}

static int
_copy_from0d(PyArrayObject *dest, PyArrayObject *src, int usecopy, int swap)
{
    char *aligned = NULL;
    char *sptr;
    int numcopies, nbytes;
    void (*myfunc)(char *, intp, char *, intp, intp, int);
    int retval = -1;
    NPY_BEGIN_THREADS_DEF;

    numcopies = PyArray_SIZE(dest);
    if (numcopies < 1) {
        return 0;
    }
    nbytes = PyArray_ITEMSIZE(src);

    if (!PyArray_ISALIGNED(src)) {
        aligned = malloc((size_t)nbytes);
        if (aligned == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(aligned, src->data, (size_t) nbytes);
        usecopy = 1;
        sptr = aligned;
    }
    else {
        sptr = src->data;
    }
    if (PyArray_SAFEALIGNEDCOPY(dest)) {
        myfunc = _strided_byte_copy;
    }
    else if (usecopy) {
        myfunc = _unaligned_strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_move;
    }

    if ((dest->nd < 2) || PyArray_ISONESEGMENT(dest)) {
        char *dptr;
        intp dstride;

        dptr = dest->data;
        if (dest->nd == 1) {
            dstride = dest->strides[0];
        }
        else {
            dstride = nbytes;
        }

        /* Refcount note: src and dest may have different sizes */
        PyArray_INCREF(src);
        PyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        myfunc(dptr, dstride, sptr, 0, numcopies, (int) nbytes);
        if (swap) {
            _strided_byte_swap(dptr, dstride, numcopies, (int) nbytes);
        }
        NPY_END_THREADS;
        PyArray_INCREF(dest);
        PyArray_XDECREF(src);
    }
    else {
        PyArrayIterObject *dit;
        int axis = -1;

        dit = (PyArrayIterObject *)
            PyArray_IterAllButAxis((PyObject *)dest, &axis);
        if (dit == NULL) {
            goto finish;
        }
        /* Refcount note: src and dest may have different sizes */
        PyArray_INCREF(src);
        PyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        while(dit->index < dit->size) {
            myfunc(dit->dataptr, PyArray_STRIDE(dest, axis), sptr, 0,
                    PyArray_DIM(dest, axis), nbytes);
            if (swap) {
                _strided_byte_swap(dit->dataptr, PyArray_STRIDE(dest, axis),
                        PyArray_DIM(dest, axis), nbytes);
            }
            PyArray_ITER_NEXT(dit);
        }
        NPY_END_THREADS;
        PyArray_INCREF(dest);
        PyArray_XDECREF(src);
        Py_DECREF(dit);
    }
    retval = 0;

finish:
    if (aligned != NULL) {
        free(aligned);
    }
    return retval;
}

/*
 * Special-case of PyArray_CopyInto when dst is 1-d
 * and contiguous (and aligned).
 * PyArray_CopyInto requires broadcastable arrays while
 * this one is a flattening operation...
 */
NPY_NO_EXPORT int
_flat_copyinto(PyObject *dst, PyObject *src, NPY_ORDER order)
{
    PyArrayIterObject *it;
    PyObject *orig_src;
    void (*myfunc)(char *, intp, char *, intp, intp, int);
    char *dptr;
    int axis;
    int elsize;
    intp nbytes;
    NPY_BEGIN_THREADS_DEF;


    orig_src = src;
    if (PyArray_NDIM(src) == 0) {
        /* Refcount note: src and dst have the same size */
        PyArray_INCREF((PyArrayObject *)src);
        PyArray_XDECREF((PyArrayObject *)dst);
        NPY_BEGIN_THREADS;
        memcpy(PyArray_BYTES(dst), PyArray_BYTES(src),
                PyArray_ITEMSIZE(src));
        NPY_END_THREADS;
        return 0;
    }

    axis = PyArray_NDIM(src)-1;

    if (order == PyArray_FORTRANORDER) {
        if (PyArray_NDIM(src) <= 2) {
            axis = 0;
        }
        /* fall back to a more general method */
        else {
            src = PyArray_Transpose((PyArrayObject *)orig_src, NULL);
        }
    }

    it = (PyArrayIterObject *)PyArray_IterAllButAxis(src, &axis);
    if (it == NULL) {
        if (src != orig_src) {
            Py_DECREF(src);
        }
        return -1;
    }

    if (PyArray_SAFEALIGNEDCOPY(src)) {
        myfunc = _strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_copy;
    }

    dptr = PyArray_BYTES(dst);
    elsize = PyArray_ITEMSIZE(dst);
    nbytes = elsize * PyArray_DIM(src, axis);

    /* Refcount note: src and dst have the same size */
    PyArray_INCREF((PyArrayObject *)src);
    PyArray_XDECREF((PyArrayObject *)dst);
    NPY_BEGIN_THREADS;
    while(it->index < it->size) {
        myfunc(dptr, elsize, it->dataptr, PyArray_STRIDE(src,axis),
                PyArray_DIM(src,axis), elsize);
        dptr += nbytes;
        PyArray_ITER_NEXT(it);
    }
    NPY_END_THREADS;

    if (src != orig_src) {
        Py_DECREF(src);
    }
    Py_DECREF(it);
    return 0;
}


static int
_copy_from_same_shape(PyArrayObject *dest, PyArrayObject *src,
                      void (*myfunc)(char *, intp, char *, intp, intp, int),
                      int swap)
{
    int maxaxis = -1, elsize;
    intp maxdim;
    PyArrayIterObject *dit, *sit;
    NPY_BEGIN_THREADS_DEF;

    dit = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)dest, &maxaxis);
    sit = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)src, &maxaxis);

    maxdim = dest->dimensions[maxaxis];

    if ((dit == NULL) || (sit == NULL)) {
        Py_XDECREF(dit);
        Py_XDECREF(sit);
        return -1;
    }
    elsize = PyArray_ITEMSIZE(dest);

    /* Refcount note: src and dst have the same size */
    PyArray_INCREF(src);
    PyArray_XDECREF(dest);

    NPY_BEGIN_THREADS;
    while(dit->index < dit->size) {
        /* strided copy of elsize bytes */
        myfunc(dit->dataptr, dest->strides[maxaxis],
                sit->dataptr, src->strides[maxaxis],
                maxdim, elsize);
        if (swap) {
            _strided_byte_swap(dit->dataptr,
                    dest->strides[maxaxis],
                    dest->dimensions[maxaxis],
                    elsize);
        }
        PyArray_ITER_NEXT(dit);
        PyArray_ITER_NEXT(sit);
    }
    NPY_END_THREADS;

    Py_DECREF(sit);
    Py_DECREF(dit);
    return 0;
}

static int
_broadcast_copy(PyArrayObject *dest, PyArrayObject *src,
                void (*myfunc)(char *, intp, char *, intp, intp, int),
                int swap)
{
    int elsize;
    PyArrayMultiIterObject *multi;
    int maxaxis; intp maxdim;
    NPY_BEGIN_THREADS_DEF;

    elsize = PyArray_ITEMSIZE(dest);
    multi = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, dest, src);
    if (multi == NULL) {
        return -1;
    }

    if (multi->size != PyArray_SIZE(dest)) {
        PyErr_SetString(PyExc_ValueError,
                "array dimensions are not "\
                "compatible for copy");
        Py_DECREF(multi);
        return -1;
    }

    maxaxis = PyArray_RemoveSmallest(multi);
    if (maxaxis < 0) {
        /*
         * copy 1 0-d array to another
         * Refcount note: src and dst have the same size
         */
        PyArray_INCREF(src);
        PyArray_XDECREF(dest);
        memcpy(dest->data, src->data, elsize);
        if (swap) {
            byte_swap_vector(dest->data, 1, elsize);
        }
        return 0;
    }
    maxdim = multi->dimensions[maxaxis];

    /*
     * Increment the source and decrement the destination
     * reference counts
     *
     * Refcount note: src and dest may have different sizes
     */
    PyArray_INCREF(src);
    PyArray_XDECREF(dest);

    NPY_BEGIN_THREADS;
    while(multi->index < multi->size) {
        myfunc(multi->iters[0]->dataptr,
                multi->iters[0]->strides[maxaxis],
                multi->iters[1]->dataptr,
                multi->iters[1]->strides[maxaxis],
                maxdim, elsize);
        if (swap) {
            _strided_byte_swap(multi->iters[0]->dataptr,
                    multi->iters[0]->strides[maxaxis],
                    maxdim, elsize);
        }
        PyArray_MultiIter_NEXT(multi);
    }
    NPY_END_THREADS;

    PyArray_INCREF(dest);
    PyArray_XDECREF(src);

    Py_DECREF(multi);
    return 0;
}

/* If destination is not the right type, then src
   will be cast to destination -- this requires
   src and dest to have the same shape
*/

/* Requires arrays to have broadcastable shapes

   The arrays are assumed to have the same number of elements
   They can be different sizes and have different types however.
*/

static int
_array_copy_into(PyArrayObject *dest, PyArrayObject *src, int usecopy)
{
    int swap;
    void (*myfunc)(char *, intp, char *, intp, intp, int);
    int simple;
    int same;
    NPY_BEGIN_THREADS_DEF;


    if (!PyArray_EquivArrTypes(dest, src)) {
        return PyArray_CastTo(dest, src);
    }
    if (!PyArray_ISWRITEABLE(dest)) {
        PyErr_SetString(PyExc_RuntimeError,
                "cannot write to array");
        return -1;
    }
    same = PyArray_SAMESHAPE(dest, src);
    simple = same && ((PyArray_ISCARRAY_RO(src) && PyArray_ISCARRAY(dest)) ||
            (PyArray_ISFARRAY_RO(src) && PyArray_ISFARRAY(dest)));

    if (simple) {
        /* Refcount note: src and dest have the same size */
        PyArray_INCREF(src);
        PyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        if (usecopy) {
            memcpy(dest->data, src->data, PyArray_NBYTES(dest));
        }
        else {
            memmove(dest->data, src->data, PyArray_NBYTES(dest));
        }
        NPY_END_THREADS;
        return 0;
    }

    swap = PyArray_ISNOTSWAPPED(dest) != PyArray_ISNOTSWAPPED(src);

    if (src->nd == 0) {
        return _copy_from0d(dest, src, usecopy, swap);
    }

    if (PyArray_SAFEALIGNEDCOPY(dest) && PyArray_SAFEALIGNEDCOPY(src)) {
        myfunc = _strided_byte_copy;
    }
    else if (usecopy) {
        myfunc = _unaligned_strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_move;
    }
    /*
     * Could combine these because _broadcasted_copy would work as well.
     * But, same-shape copying is so common we want to speed it up.
     */
    if (same) {
        return _copy_from_same_shape(dest, src, myfunc, swap);
    }
    else {
        return _broadcast_copy(dest, src, myfunc, swap);
    }
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 */
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dest, PyArrayObject *src)
{
    int elsize, simple;
    PyArrayIterObject *idest, *isrc;
    void (*myfunc)(char *, intp, char *, intp, intp, int);
    NPY_BEGIN_THREADS_DEF;

    if (!PyArray_EquivArrTypes(dest, src)) {
        return PyArray_CastAnyTo(dest, src);
    }
    if (!PyArray_ISWRITEABLE(dest)) {
        PyErr_SetString(PyExc_RuntimeError,
                "cannot write to array");
        return -1;
    }
    if (PyArray_SIZE(dest) != PyArray_SIZE(src)) {
        PyErr_SetString(PyExc_ValueError,
                "arrays must have the same number of elements"
                " for copy");
        return -1;
    }

    simple = ((PyArray_ISCARRAY_RO(src) && PyArray_ISCARRAY(dest)) ||
            (PyArray_ISFARRAY_RO(src) && PyArray_ISFARRAY(dest)));
    if (simple) {
        /* Refcount note: src and dest have the same size */
        PyArray_INCREF(src);
        PyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        memcpy(dest->data, src->data, PyArray_NBYTES(dest));
        NPY_END_THREADS;
        return 0;
    }

    if (PyArray_SAMESHAPE(dest, src)) {
        int swap;

        if (PyArray_SAFEALIGNEDCOPY(dest) && PyArray_SAFEALIGNEDCOPY(src)) {
            myfunc = _strided_byte_copy;
        }
        else {
            myfunc = _unaligned_strided_byte_copy;
        }
        swap = PyArray_ISNOTSWAPPED(dest) != PyArray_ISNOTSWAPPED(src);
        return _copy_from_same_shape(dest, src, myfunc, swap);
    }

    /* Otherwise we have to do an iterator-based copy */
    idest = (PyArrayIterObject *)PyArray_IterNew((PyObject *)dest);
    if (idest == NULL) {
        return -1;
    }
    isrc = (PyArrayIterObject *)PyArray_IterNew((PyObject *)src);
    if (isrc == NULL) {
        Py_DECREF(idest);
        return -1;
    }
    elsize = dest->descr->elsize;
    /* Refcount note: src and dest have the same size */
    PyArray_INCREF(src);
    PyArray_XDECREF(dest);
    NPY_BEGIN_THREADS;
    while(idest->index < idest->size) {
        memcpy(idest->dataptr, isrc->dataptr, elsize);
        PyArray_ITER_NEXT(idest);
        PyArray_ITER_NEXT(isrc);
    }
    NPY_END_THREADS;
    Py_DECREF(idest);
    Py_DECREF(isrc);
    return 0;
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dest, PyArrayObject *src)
{
    return _array_copy_into(dest, src, 1);
}


/*NUMPY_API
 * Move the memory of one array into another.
 */
NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dest, PyArrayObject *src)
{
    return _array_copy_into(dest, src, 0);
}


/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object)
{
    PyArrayObject *src;
    PyObject *r;
    int ret;

    /*
     * Special code to mimic Numeric behavior for
     * character arrays.
     */
    if (dest->descr->type == PyArray_CHARLTR && dest->nd > 0 \
        && PyString_Check(src_object)) {
        intp n_new, n_old;
        char *new_string;
        PyObject *tmp;

        n_new = dest->dimensions[dest->nd-1];
        n_old = PyString_Size(src_object);
        if (n_new > n_old) {
            new_string = (char *)malloc(n_new);
            memmove(new_string, PyString_AS_STRING(src_object), n_old);
            memset(new_string + n_old, ' ', n_new - n_old);
            tmp = PyString_FromStringAndSize(new_string, n_new);
            free(new_string);
            src_object = tmp;
        }
    }

    if (PyArray_Check(src_object)) {
        src = (PyArrayObject *)src_object;
        Py_INCREF(src);
    }
    else if (!PyArray_IsScalar(src_object, Generic) &&
             PyArray_HasArrayInterface(src_object, r)) {
        src = (PyArrayObject *)r;
    }
    else {
        PyArray_Descr* dtype;
        dtype = dest->descr;
        Py_INCREF(dtype);
        src = (PyArrayObject *)PyArray_FromAny(src_object, dtype, 0,
                                               dest->nd,
                                               FORTRAN_IF(dest),
                                               NULL);
    }
    if (src == NULL) {
        return -1;
    }

    ret = PyArray_MoveInto(dest, src);
    Py_DECREF(src);
    return ret;
}


/* These are also old calls (should use PyArray_NewFromDescr) */

/* They all zero-out the memory as previously done */

/* steals reference to descr -- and enforces native byteorder on it.*/
/*NUMPY_API
  Like FromDimsAndData but uses the Descr structure instead of typecode
  as input.
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDimsAndDataAndDescr(int nd, int *d,
                                PyArray_Descr *descr,
                                char *data)
{
    PyObject *ret;
    int i;
    intp newd[MAX_DIMS];
    char msg[] = "PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    if (!PyArray_ISNBO(descr->byteorder))
        descr->byteorder = '=';
    for (i = 0; i < nd; i++) {
        newd[i] = (intp) d[i];
    }
    ret = PyArray_NewFromDescr(&PyArray_Type, descr,
                               nd, newd,
                               NULL, data,
                               (data ? CARRAY : 0), NULL);
    return ret;
}

/*NUMPY_API
  Construct an empty array from dimensions and typenum
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDims(int nd, int *d, int type)
{
    PyObject *ret;
    char msg[] = "PyArray_FromDims: use PyArray_SimpleNew.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    ret = PyArray_FromDimsAndDataAndDescr(nd, d,
                                          PyArray_DescrFromType(type),
                                          NULL);
    /*
     * Old FromDims set memory to zero --- some algorithms
     * relied on that.  Better keep it the same. If
     * Object type, then it's already been set to zero, though.
     */
    if (ret && (PyArray_DESCR(ret)->type_num != PyArray_OBJECT)) {
        memset(PyArray_DATA(ret), 0, PyArray_NBYTES(ret));
    }
    return ret;
}

/* end old calls */


/*NUMPY_API
  Copy an array.
*/
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *m1, NPY_ORDER fortran)
{
    PyArrayObject *ret;
    if (fortran == PyArray_ANYORDER)
        fortran = PyArray_ISFORTRAN(m1);

    Py_INCREF(m1->descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(m1->ob_type,
                                                m1->descr,
                                                m1->nd,
                                                m1->dimensions,
                                                NULL, NULL,
                                                fortran,
                                                (PyObject *)m1);
    if (ret == NULL) {
        return NULL;
    }
    if (PyArray_CopyInto(ret, m1) == -1) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject *)ret;
}

/* Does nothing with descr (cannot be NULL) */
/*NUMPY_API
  Get scalar-equivalent to a region of memory described by a descriptor.
*/
NPY_NO_EXPORT PyObject *
PyArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base)
{
    PyTypeObject *type;
    PyObject *obj;
    void *destptr;
    PyArray_CopySwapFunc *copyswap;
    int type_num;
    int itemsize;
    int swap;

    type_num = descr->type_num;
    if (type_num == PyArray_BOOL) {
        PyArrayScalar_RETURN_BOOL_FROM_LONG(*(Bool*)data);
    }
    else if (PyDataType_FLAGCHK(descr, NPY_USE_GETITEM)) {
        return descr->f->getitem(data, base);
    }
    itemsize = descr->elsize;
    copyswap = descr->f->copyswap;
    type = descr->typeobj;
    swap = !PyArray_ISNBO(descr->byteorder);
    if PyTypeNum_ISSTRING(type_num) { /* Eliminate NULL bytes */
            char *dptr = data;

            dptr += itemsize - 1;
            while(itemsize && *dptr-- == 0) {
                itemsize--;
            }
            if (type_num == PyArray_UNICODE && itemsize) {
                /* make sure itemsize is a multiple of 4 */
                /* so round up to nearest multiple */
                itemsize = (((itemsize-1) >> 2) + 1) << 2;
            }
        }
    if (type->tp_itemsize != 0) { /* String type */
        obj = type->tp_alloc(type, itemsize);
    }
    else {
        obj = type->tp_alloc(type, 0);
    }
    if (obj == NULL) {
        return NULL;
    }
    if PyTypeNum_ISFLEXIBLE(type_num) {
            if (type_num == PyArray_STRING) {
                destptr = PyString_AS_STRING(obj);
                ((PyStringObject *)obj)->ob_shash = -1;
                ((PyStringObject *)obj)->ob_sstate =    \
                    SSTATE_NOT_INTERNED;
                memcpy(destptr, data, itemsize);
                return obj;
            }
            else if (type_num == PyArray_UNICODE) {
                PyUnicodeObject *uni = (PyUnicodeObject*)obj;
                size_t length = itemsize >> 2;
#ifndef Py_UNICODE_WIDE
                char *buffer;
                int alloc = 0;
                length *= 2;
#endif
                /* Need an extra slot and need to use
                   Python memory manager */
                uni->str = NULL;
                destptr = PyMem_NEW(Py_UNICODE,length+1);
                if (destptr == NULL) {
                    Py_DECREF(obj);
                    return PyErr_NoMemory();
                }
                uni->str = (Py_UNICODE *)destptr;
                uni->str[0] = 0;
                uni->str[length] = 0;
                uni->length = length;
                uni->hash = -1;
                uni->defenc = NULL;
#ifdef Py_UNICODE_WIDE
                memcpy(destptr, data, itemsize);
                if (swap) {
                    byte_swap_vector(destptr, length, 4);
                }
#else
                /* need aligned data buffer */
                if ((swap) || ((((intp)data) % descr->alignment) != 0)) {
                    buffer = _pya_malloc(itemsize);
                    if (buffer == NULL) {
                        return PyErr_NoMemory();
                    }
                    alloc = 1;
                    memcpy(buffer, data, itemsize);
                    if (swap) {
                        byte_swap_vector(buffer, itemsize >> 2, 4);
                    }
                }
                else {
                    buffer = data;
                }

                /* Allocated enough for 2-characters per itemsize.
                   Now convert from the data-buffer
                */
                length = PyUCS2Buffer_FromUCS4(uni->str,
                                               (PyArray_UCS4 *)buffer,
                                               itemsize >> 2);
                if (alloc) {
                    _pya_free(buffer);
                }
                /* Resize the unicode result */
                if (MyPyUnicode_Resize(uni, length) < 0) {
                    Py_DECREF(obj);
                    return NULL;
                }
#endif
                return obj;
            }
            else {
                PyVoidScalarObject *vobj = (PyVoidScalarObject *)obj;
                vobj->base = NULL;
                vobj->descr = descr;
                Py_INCREF(descr);
                vobj->obval = NULL;
                vobj->ob_size = itemsize;
                vobj->flags = BEHAVED | OWNDATA;
                swap = 0;
                if (descr->names) {
                    if (base) {
                        Py_INCREF(base);
                        vobj->base = base;
                        vobj->flags = PyArray_FLAGS(base);
                        vobj->flags &= ~OWNDATA;
                        vobj->obval = data;
                        return obj;
                    }
                }
                destptr = PyDataMem_NEW(itemsize);
                if (destptr == NULL) {
                    Py_DECREF(obj);
                    return PyErr_NoMemory();
                }
                vobj->obval = destptr;
            }
        }
    else {
        destptr = scalar_value(obj, descr);
    }
    /* copyswap for OBJECT increments the reference count */
    copyswap(destptr, data, swap, base);
    return obj;
}

/* returns an Array-Scalar Object of the type of arr
   from the given pointer to memory -- main Scalar creation function
   default new method calls this.
*/

/* Ideally, here the descriptor would contain all the information needed.
   So, that we simply need the data and the descriptor, and perhaps
   a flag
*/


/* Return Array Scalar if 0-d array object is encountered */

/*NUMPY_API
  Return either an array or the appropriate Python object if the array
  is 0d and matches a Python type.
*/
NPY_NO_EXPORT PyObject *
PyArray_Return(PyArrayObject *mp)
{

    if (mp == NULL) {
        return NULL;
    }
    if (PyErr_Occurred()) {
        Py_XDECREF(mp);
        return NULL;
    }
    if (!PyArray_Check(mp)) {
        return (PyObject *)mp;
    }
    if (mp->nd == 0) {
        PyObject *ret;
        ret = PyArray_ToScalar(mp->data, mp);
        Py_DECREF(mp);
        return ret;
    }
    else {
        return (PyObject *)mp;
    }
}


/*NUMPY_API
  Initialize arrfuncs to NULL
*/
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < PyArray_NTYPES; i++) {
        f->cast[i] = NULL;
    }
    f->getitem = NULL;
    f->setitem = NULL;
    f->copyswapn = NULL;
    f->copyswap = NULL;
    f->compare = NULL;
    f->argmax = NULL;
    f->dotfunc = NULL;
    f->scanfunc = NULL;
    f->fromstr = NULL;
    f->nonzero = NULL;
    f->fill = NULL;
    f->fillwithscalar = NULL;
    for(i = 0; i < PyArray_NSORTS; i++) {
        f->sort[i] = NULL;
        f->argsort[i] = NULL;
    }
    f->castdict = NULL;
    f->scalarkind = NULL;
    f->cancastscalarkindto = NULL;
    f->cancastto = NULL;
}

static Bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = PyArray_ITEMSIZE(arr);
    char *ptr = ip;
    while (elsize--) {
        if (*ptr++ != 0) {
            return TRUE;
        }
    }
    return FALSE;
}

static void
_default_copyswapn(void *dst, npy_intp dstride, void *src,
		   npy_intp sstride, npy_intp n, int swap, void *arr)
{
    npy_intp i;
    PyArray_CopySwapFunc *copyswap;
    char *dstptr = dst;
    char *srcptr = src;

    copyswap = PyArray_DESCR(arr)->f->copyswap;

    for (i = 0; i < n; i++) {
	copyswap(dstptr, srcptr, swap, arr);
	dstptr += dstride;
	srcptr += sstride;
    }
}

/*
  Given a string return the type-number for
  the data-type with that string as the type-object name.
  Returns PyArray_NOTYPE without setting an error if no type can be
  found.  Only works for user-defined data-types.
*/

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_TypeNumFromName(char *str)
{
    int i;
    PyArray_Descr *descr;

    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = userdescrs[i];
        if (strcmp(descr->typeobj->tp_name, str) == 0) {
            return descr->type_num;
        }
    }
    return PyArray_NOTYPE;
}

/*
  returns typenum to associate with this type >=PyArray_USERDEF.
  needs the userdecrs table and PyArray_NUMUSER variables
  defined in arraytypes.inc
*/
/*NUMPY_API
  Register Data type
  Does not change the reference count of descr
*/
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr)
{
    PyArray_Descr *descr2;
    int typenum;
    int i;
    PyArray_ArrFuncs *f;

    /* See if this type is already registered */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr2 = userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = PyArray_USERDEF + NPY_NUMUSERTYPES;
    descr->type_num = typenum;
    if (descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "cannot register a" \
                        "flexible data-type");
        return -1;
    }
    f = descr->f;
    if (f->nonzero == NULL) {
        f->nonzero = _default_nonzero;
    }
    if (f->copyswapn == NULL) {
	f->copyswapn = _default_copyswapn;
    }
    if (f->copyswap == NULL || f->getitem == NULL ||
        f->setitem == NULL) {
	PyErr_SetString(PyExc_ValueError, "a required array function"	\
                        " is missing.");
        return -1;
    }
    if (descr->typeobj == NULL) {
        PyErr_SetString(PyExc_ValueError, "missing typeobject");
        return -1;
    }
    userdescrs = realloc(userdescrs,
                         (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (userdescrs == NULL) {
        PyErr_SetString(PyExc_MemoryError, "RegisterDataType");
        return -1;
    }
    userdescrs[NPY_NUMUSERTYPES++] = descr;
    return typenum;
}

/*NUMPY_API
  Register Casting Function
  Replaces any function currently stored.
*/
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc)
{
    PyObject *cobj, *key;
    int ret;

    if (totype < PyArray_NTYPES) {
        descr->f->cast[totype] = castfunc;
        return 0;
    }
    if (!PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_TypeError, "invalid type number.");
        return -1;
    }
    if (descr->f->castdict == NULL) {
        descr->f->castdict = PyDict_New();
        if (descr->f->castdict == NULL) {
            return -1;
        }
    }
    key = PyInt_FromLong(totype);
    if (PyErr_Occurred()) {
        return -1;
    }
    cobj = PyCObject_FromVoidPtr((void *)castfunc, NULL);
    if (cobj == NULL) {
        Py_DECREF(key);
        return -1;
    }
    ret = PyDict_SetItem(descr->f->castdict, key, cobj);
    Py_DECREF(key);
    Py_DECREF(cobj);
    return ret;
}

static int *
_append_new(int *types, int insert)
{
    int n = 0;
    int *newtypes;

    while (types[n] != PyArray_NOTYPE) {
        n++;
    }
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    newtypes[n] = insert;
    newtypes[n + 1] = PyArray_NOTYPE;
    return newtypes;
}

/*NUMPY_API
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar)
{
    if (scalar == PyArray_NOSCALAR) {
        /*
         * register with cancastto
         * These lists won't be freed once created
         * -- they become part of the data-type
         */
        if (descr->f->cancastto == NULL) {
            descr->f->cancastto = (int *)malloc(1*sizeof(int));
            descr->f->cancastto[0] = PyArray_NOTYPE;
        }
        descr->f->cancastto = _append_new(descr->f->cancastto,
                                          totype);
    }
    else {
        /* register with cancastscalarkindto */
        if (descr->f->cancastscalarkindto == NULL) {
            int i;
            descr->f->cancastscalarkindto =
                (int **)malloc(PyArray_NSCALARKINDS* sizeof(int*));
            for (i = 0; i < PyArray_NSCALARKINDS; i++) {
                descr->f->cancastscalarkindto[i] = NULL;
            }
        }
        if (descr->f->cancastscalarkindto[scalar] == NULL) {
            descr->f->cancastscalarkindto[scalar] =
                (int *)malloc(1*sizeof(int));
            descr->f->cancastscalarkindto[scalar][0] =
                PyArray_NOTYPE;
        }
        descr->f->cancastscalarkindto[scalar] =
            _append_new(descr->f->cancastscalarkindto[scalar], totype);
    }
    return 0;
}


/* XXX: FIXME --- add ordering argument to
   Allow Fortran ordering on write
   This will need the addition of a Fortran-order iterator.
 */

/*NUMPY_API
  To File
*/
NPY_NO_EXPORT int
PyArray_ToFile(PyArrayObject *self, FILE *fp, char *sep, char *format)
{
    intp size;
    intp n, n2;
    size_t n3, n4;
    PyArrayIterObject *it;
    PyObject *obj, *strobj, *tupobj;

    n3 = (sep ? strlen((const char *)sep) : 0);
    if (n3 == 0) {
        /* binary data */
        if (PyDataType_FLAGCHK(self->descr, NPY_LIST_PICKLE)) {
            PyErr_SetString(PyExc_ValueError, "cannot write " \
                    "object arrays to a file in "   \
                    "binary mode");
            return -1;
        }

        if (PyArray_ISCONTIGUOUS(self)) {
            size = PyArray_SIZE(self);
            NPY_BEGIN_ALLOW_THREADS;
            n = fwrite((const void *)self->data,
                    (size_t) self->descr->elsize,
                    (size_t) size, fp);
            NPY_END_ALLOW_THREADS;
            if (n < size) {
                PyErr_Format(PyExc_ValueError,
                        "%ld requested and %ld written",
                        (long) size, (long) n);
                return -1;
            }
        }
        else {
            NPY_BEGIN_THREADS_DEF;

            it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)self);
            NPY_BEGIN_THREADS;
            while (it->index < it->size) {
                if (fwrite((const void *)it->dataptr,
                            (size_t) self->descr->elsize,
                            1, fp) < 1) {
                    NPY_END_THREADS;
                    PyErr_Format(PyExc_IOError,
                            "problem writing element"\
                            " %"INTP_FMT" to file",
                            it->index);
                    Py_DECREF(it);
                    return -1;
                }
                PyArray_ITER_NEXT(it);
            }
            NPY_END_THREADS;
            Py_DECREF(it);
        }
    }
    else {
        /*
         * text data
         */

        it = (PyArrayIterObject *)
            PyArray_IterNew((PyObject *)self);
        n4 = (format ? strlen((const char *)format) : 0);
        while (it->index < it->size) {
            obj = self->descr->f->getitem(it->dataptr, self);
            if (obj == NULL) {
                Py_DECREF(it);
                return -1;
            }
            if (n4 == 0) {
                /*
                 * standard writing
                 */
                strobj = PyObject_Str(obj);
                Py_DECREF(obj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
            else {
                /*
                 * use format string
                 */
                tupobj = PyTuple_New(1);
                if (tupobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
                PyTuple_SET_ITEM(tupobj,0,obj);
                obj = PyString_FromString((const char *)format);
                if (obj == NULL) {
                    Py_DECREF(tupobj);
                    Py_DECREF(it);
                    return -1;
                }
                strobj = PyString_Format(obj, tupobj);
                Py_DECREF(obj);
                Py_DECREF(tupobj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
            NPY_BEGIN_ALLOW_THREADS;
            n2 = PyString_GET_SIZE(strobj);
            n = fwrite(PyString_AS_STRING(strobj), 1, n2, fp);
            NPY_END_ALLOW_THREADS;
            if (n < n2) {
                PyErr_Format(PyExc_IOError,
                        "problem writing element %"INTP_FMT\
                        " to file", it->index);
                Py_DECREF(strobj);
                Py_DECREF(it);
                return -1;
            }
            /* write separator for all but last one */
            if (it->index != it->size-1) {
                if (fwrite(sep, 1, n3, fp) < n3) {
                    PyErr_Format(PyExc_IOError,
                            "problem writing "\
                            "separator to file");
                    Py_DECREF(strobj);
                    Py_DECREF(it);
                    return -1;
                }
            }
            Py_DECREF(strobj);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
 * To List
 */
NPY_NO_EXPORT PyObject *
PyArray_ToList(PyArrayObject *self)
{
    PyObject *lp;
    PyArrayObject *v;
    intp sz, i;

    if (!PyArray_Check(self)) {
        return (PyObject *)self;
    }
    if (self->nd == 0) {
        return self->descr->f->getitem(self->data,self);
    }

    sz = self->dimensions[0];
    lp = PyList_New(sz);
    for (i = 0; i < sz; i++) {
	v = (PyArrayObject *)array_big_item(self, i);
	if (PyArray_Check(v) && (v->nd >= self->nd)) {
	    PyErr_SetString(PyExc_RuntimeError,
			    "array_item not returning smaller-"	\
			    "dimensional array");
	    Py_DECREF(v);
	    Py_DECREF(lp);
	    return NULL;
	}
        PyList_SetItem(lp, i, PyArray_ToList(v));
        Py_DECREF(v);
    }
    return lp;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_ToString(PyArrayObject *self, NPY_ORDER order)
{
    intp numbytes;
    intp index;
    char *dptr;
    int elsize;
    PyObject *ret;
    PyArrayIterObject *it;

    if (order == NPY_ANYORDER)
        order = PyArray_ISFORTRAN(self);

    /*        if (PyArray_TYPE(self) == PyArray_OBJECT) {
              PyErr_SetString(PyExc_ValueError, "a string for the data" \
              "in an object array is not appropriate");
              return NULL;
              }
    */

    numbytes = PyArray_NBYTES(self);
    if ((PyArray_ISCONTIGUOUS(self) && (order == NPY_CORDER))
        || (PyArray_ISFORTRAN(self) && (order == NPY_FORTRANORDER))) {
        ret = PyString_FromStringAndSize(self->data, (Py_ssize_t) numbytes);
    }
    else {
        PyObject *new;
        if (order == NPY_FORTRANORDER) {
            /* iterators are always in C-order */
            new = PyArray_Transpose(self, NULL);
            if (new == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(self);
            new = (PyObject *)self;
        }
        it = (PyArrayIterObject *)PyArray_IterNew(new);
        Py_DECREF(new);
        if (it == NULL) {
            return NULL;
        }
        ret = PyString_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            Py_DECREF(it);
            return NULL;
        }
        dptr = PyString_AS_STRING(ret);
        index = it->size;
        elsize = self->descr->elsize;
        while (index--) {
            memcpy(dptr, it->dataptr, elsize);
            dptr += elsize;
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return ret;
}


/*********************** end C-API functions **********************/


/* array object functions */

static void
array_dealloc(PyArrayObject *self) {

    if (self->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    if (self->base) {
        /*
         * UPDATEIFCOPY means that base points to an
         * array that should be updated with the contents
         * of this array upon destruction.
         * self->base->flags must have been WRITEABLE
         * (checked previously) and it was locked here
         * thus, unlock it.
         */
        if (self->flags & UPDATEIFCOPY) {
            ((PyArrayObject *)self->base)->flags |= WRITEABLE;
            Py_INCREF(self); /* hold on to self in next call */
            if (PyArray_CopyAnyInto((PyArrayObject *)self->base, self) < 0) {
                PyErr_Print();
                PyErr_Clear();
            }
            /*
             * Don't need to DECREF -- because we are deleting
             *self already...
             */
        }
        /*
         * In any case base is pointing to something that we need
         * to DECREF -- either a view or a buffer object
         */
        Py_DECREF(self->base);
    }

    if ((self->flags & OWNDATA) && self->data) {
        /* Free internal references if an Object array */
        if (PyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            Py_INCREF(self); /*hold on to self */
            PyArray_XDECREF(self);
            /*
             * Don't need to DECREF -- because we are deleting
             * self already...
             */
        }
        PyDataMem_FREE(self->data);
    }

    PyDimMem_FREE(self->dimensions);
    Py_DECREF(self->descr);
    self->ob_type->tp_free((PyObject *)self);
}

/*************************************************************************
 ****************   Implement Buffer Protocol ****************************
 *************************************************************************/

/* removed multiple segment interface */

static Py_ssize_t
array_getsegcount(PyArrayObject *self, Py_ssize_t *lenp)
{
    if (lenp) {
        *lenp = PyArray_NBYTES(self);
    }
    if (PyArray_ISONESEGMENT(self)) {
        return 1;
    }
    if (lenp) {
        *lenp = 0;
    }
    return 0;
}

static Py_ssize_t
array_getreadbuf(PyArrayObject *self, Py_ssize_t segment, void **ptrptr)
{
    if (segment != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "accessing non-existing array segment");
        return -1;
    }
    if (PyArray_ISONESEGMENT(self)) {
        *ptrptr = self->data;
        return PyArray_NBYTES(self);
    }
    PyErr_SetString(PyExc_ValueError, "array is not a single segment");
    *ptrptr = NULL;
    return -1;
}


static Py_ssize_t
array_getwritebuf(PyArrayObject *self, Py_ssize_t segment, void **ptrptr)
{
    if (PyArray_CHKFLAGS(self, WRITEABLE)) {
        return array_getreadbuf(self, segment, (void **) ptrptr);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "array cannot be "
                        "accessed as a writeable buffer");
        return -1;
    }
}

static Py_ssize_t
array_getcharbuf(PyArrayObject *self, Py_ssize_t segment, constchar **ptrptr)
{
    return array_getreadbuf(self, segment, (void **) ptrptr);
}

static PyBufferProcs array_as_buffer = {
#if PY_VERSION_HEX >= 0x02050000
    (readbufferproc)array_getreadbuf,       /*bf_getreadbuffer*/
    (writebufferproc)array_getwritebuf,     /*bf_getwritebuffer*/
    (segcountproc)array_getsegcount,        /*bf_getsegcount*/
    (charbufferproc)array_getcharbuf,       /*bf_getcharbuffer*/
#else
    (getreadbufferproc)array_getreadbuf,    /*bf_getreadbuffer*/
    (getwritebufferproc)array_getwritebuf,  /*bf_getwritebuffer*/
    (getsegcountproc)array_getsegcount,     /*bf_getsegcount*/
    (getcharbufferproc)array_getcharbuf,    /*bf_getcharbuffer*/
#endif
};

/****************** End of Buffer Protocol *******************************/


/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

NPY_NO_EXPORT NumericOps n_ops; /* NB: static objects initialized to zero */

/* Dictionary can contain any of the numeric operations, by name.
   Those not present will not be changed
*/

#define SET(op)   temp=PyDict_GetItemString(dict, #op); \
    if (temp != NULL) {                                 \
        if (!(PyCallable_Check(temp))) return -1;       \
        Py_XDECREF(n_ops.op);                           \
        n_ops.op = temp;                                \
    }


/*NUMPY_API
  Set internal structure with number functions that all arrays will use
*/
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(ones_like);
    SET(sqrt);
    SET(negative);
    SET(absolute);
    SET(invert);
    SET(left_shift);
    SET(right_shift);
    SET(bitwise_and);
    SET(bitwise_or);
    SET(bitwise_xor);
    SET(less);
    SET(less_equal);
    SET(equal);
    SET(not_equal);
    SET(greater);
    SET(greater_equal);
    SET(floor_divide);
    SET(true_divide);
    SET(logical_or);
    SET(logical_and);
    SET(floor);
    SET(ceil);
    SET(maximum);
    SET(minimum);
    SET(rint);
    SET(conjugate);
    return 0;
}

#define GET(op) if (n_ops.op &&                                         \
                    (PyDict_SetItemString(dict, #op, n_ops.op)==-1))    \
        goto fail;

/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(power);
    GET(square);
    GET(reciprocal);
    GET(ones_like);
    GET(sqrt);
    GET(negative);
    GET(absolute);
    GET(invert);
    GET(left_shift);
    GET(right_shift);
    GET(bitwise_and);
    GET(bitwise_or);
    GET(bitwise_xor);
    GET(less);
    GET(less_equal);
    GET(equal);
    GET(not_equal);
    GET(greater);
    GET(greater_equal);
    GET(floor_divide);
    GET(true_divide);
    GET(logical_or);
    GET(logical_and);
    GET(floor);
    GET(ceil);
    GET(maximum);
    GET(minimum);
    GET(rint);
    GET(conjugate);
    return dict;

 fail:
    Py_DECREF(dict);
    return NULL;
}

static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    if (rtype != PyArray_NOTYPE || out != NULL) {
        kwds = PyDict_New();
        if (rtype != PyArray_NOTYPE) {
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            if (descr) {
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    return kwds;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "reduce");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "accumulate");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OO", m1, m2);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "(O)", m1);
}

static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OOO", m1, m2, m1);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OO", m1, m1);
}

static PyObject *
array_add(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
}

static PyObject *
array_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.divide);
}

static PyObject *
array_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.remainder);
}

static int
array_power_is_scalar(PyObject *o2, double* exp)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyInt_Check(o2)) {
        *exp = (double)PyInt_AsLong(o2);
        return 1;
    }
    if (optimize_fpexps && PyFloat_Check(o2)) {
        *exp = PyFloat_AsDouble(o2);
        return 1;
    }
    if ((PyArray_IsZeroDim(o2) &&
         ((PyArray_ISINTEGER(o2) ||
           (optimize_fpexps && PyArray_ISFLOAT(o2))))) ||
        PyArray_IsScalar(o2, Integer) ||
        (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = o2->ob_type->tp_as_number->nb_float(o2);
        if (temp != NULL) {
            *exp = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            return 1;
        }
    }
#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value==NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return 0;
        }
        val = PyInt_AsSsize_t(value);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        *exp = (double) val;
        return 1;
    }
#endif
    return 0;
}

/* optimize float array or complex array to a scalar power */
static PyObject *
fast_scalar_power(PyArrayObject *a1, PyObject *o2, int inplace)
{
    double exp;

    if (PyArray_Check(a1) && array_power_is_scalar(o2, &exp)) {
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exp == 1.0) {
                /* we have to do this one special, as the
                   "copy" method of array objects isn't set
                   up early enough to be added
                   by PyArray_SetNumericOps.
                */
                if (inplace) {
                    Py_INCREF(a1);
                    return (PyObject *)a1;
                } else {
                    return PyArray_Copy(a1);
                }
            }
            else if (exp == -1.0) {
                fastop = n_ops.reciprocal;
            }
            else if (exp ==  0.0) {
                fastop = n_ops.ones_like;
            }
            else if (exp ==  0.5) {
                fastop = n_ops.sqrt;
            }
            else if (exp ==  2.0) {
                fastop = n_ops.square;
            }
            else {
                return NULL;
            }

            if (inplace) {
                return PyArray_GenericInplaceUnaryFunction(a1, fastop);
            } else {
                return PyArray_GenericUnaryFunction(a1, fastop);
            }
        }
        else if (exp==2.0) {
            fastop = n_ops.multiply;
            if (inplace) {
                return PyArray_GenericInplaceBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
            else {
                return PyArray_GenericBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
        }
    }
    return NULL;
}

static PyObject *
array_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 0);
    if (!value) {
        value = PyArray_GenericBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}


static PyObject *
array_negative(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.negative);
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.absolute);
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.invert);
}

static PyObject *
array_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.multiply);
}

static PyObject *
array_inplace_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.divide);
}

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 1);
    if (!value) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.floor_divide);
}

static PyObject *
array_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.true_divide);
}

static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.floor_divide);
}

static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.true_divide);
}

/* Array evaluates as "TRUE" if any of the elements are non-zero*/
static int
array_any_nonzero(PyArrayObject *mp)
{
    intp index;
    PyArrayIterObject *it;
    Bool anyTRUE = FALSE;

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
    if (it == NULL) {
        return anyTRUE;
    }
    index = it->size;
    while(index--) {
        if (mp->descr->f->nonzero(it->dataptr, mp)) {
            anyTRUE = TRUE;
            break;
        }
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(it);
    return anyTRUE;
}

static int
_array_nonzero(PyArrayObject *mp)
{
    intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        return mp->descr->f->nonzero(mp->data, mp);
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array " \
                        "with more than one element is ambiguous. " \
                        "Use a.any() or a.all()");
        return -1;
    }
}



static PyObject *
array_divmod(PyArrayObject *op1, PyObject *op2)
{
    PyObject *divp, *modp, *result;

    divp = array_floor_divide(op1, op2);
    if (divp == NULL) {
        return NULL;
    }
    modp = array_remainder(op1, op2);
    if (modp == NULL) {
        Py_DECREF(divp);
        return NULL;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}


static PyObject *
array_int(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be"\
                        " converted to Python scalars");
        return NULL;
    }
    pv = v->descr->f->getitem(v->data, v);
    if (pv == NULL) {
        return NULL;
    }
    if (pv->ob_type->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_int == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to int");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = pv->ob_type->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_float(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = v->descr->f->getitem(v->data, v);
    if (pv == NULL) {
        return NULL;
    }
    if (pv->ob_type->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a "\
                        "float; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_float == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to float");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_long(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = v->descr->f->getitem(v->data, v);
    if (pv->ob_type->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_long == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to long");
        return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_oct(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = v->descr->f->getitem(v->data, v);
    if (pv->ob_type->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_oct == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to oct");
        return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_hex(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = v->descr->f->getitem(v->data, v);
    if (pv->ob_type->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_hex == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to hex");
        return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
_array_copy_nice(PyArrayObject *self)
{
    return PyArray_Return((PyArrayObject *) PyArray_Copy(self));
}

#if PY_VERSION_HEX >= 0x02050000
static PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only integer arrays with "     \
                        "one element can be converted to an index");
        return NULL;
    }
    return v->descr->f->getitem(v->data, v);
}
#endif


static PyNumberMethods array_as_number = {
    (binaryfunc)array_add,                      /*nb_add*/
    (binaryfunc)array_subtract,                 /*nb_subtract*/
    (binaryfunc)array_multiply,                 /*nb_multiply*/
    (binaryfunc)array_divide,                   /*nb_divide*/
    (binaryfunc)array_remainder,                /*nb_remainder*/
    (binaryfunc)array_divmod,                   /*nb_divmod*/
    (ternaryfunc)array_power,                   /*nb_power*/
    (unaryfunc)array_negative,                  /*nb_neg*/
    (unaryfunc)_array_copy_nice,                /*nb_pos*/
    (unaryfunc)array_absolute,                  /*(unaryfunc)array_abs,*/
    (inquiry)_array_nonzero,                    /*nb_nonzero*/
    (unaryfunc)array_invert,                    /*nb_invert*/
    (binaryfunc)array_left_shift,               /*nb_lshift*/
    (binaryfunc)array_right_shift,              /*nb_rshift*/
    (binaryfunc)array_bitwise_and,              /*nb_and*/
    (binaryfunc)array_bitwise_xor,              /*nb_xor*/
    (binaryfunc)array_bitwise_or,               /*nb_or*/
    0,                                          /*nb_coerce*/
    (unaryfunc)array_int,                       /*nb_int*/
    (unaryfunc)array_long,                      /*nb_long*/
    (unaryfunc)array_float,                     /*nb_float*/
    (unaryfunc)array_oct,                       /*nb_oct*/
    (unaryfunc)array_hex,                       /*nb_hex*/

    /*
     * This code adds augmented assignment functionality
     * that was made available in Python 2.0
     */
    (binaryfunc)array_inplace_add,              /*inplace_add*/
    (binaryfunc)array_inplace_subtract,         /*inplace_subtract*/
    (binaryfunc)array_inplace_multiply,         /*inplace_multiply*/
    (binaryfunc)array_inplace_divide,           /*inplace_divide*/
    (binaryfunc)array_inplace_remainder,        /*inplace_remainder*/
    (ternaryfunc)array_inplace_power,           /*inplace_power*/
    (binaryfunc)array_inplace_left_shift,       /*inplace_lshift*/
    (binaryfunc)array_inplace_right_shift,      /*inplace_rshift*/
    (binaryfunc)array_inplace_bitwise_and,      /*inplace_and*/
    (binaryfunc)array_inplace_bitwise_xor,      /*inplace_xor*/
    (binaryfunc)array_inplace_bitwise_or,       /*inplace_or*/

    (binaryfunc)array_floor_divide,             /*nb_floor_divide*/
    (binaryfunc)array_true_divide,              /*nb_true_divide*/
    (binaryfunc)array_inplace_floor_divide,     /*nb_inplace_floor_divide*/
    (binaryfunc)array_inplace_true_divide,      /*nb_inplace_true_divide*/

#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc)array_index,                     /* nb_index */
#endif

};

/****************** End of Buffer Protocol *******************************/


/*************************************************************************
 ****************   Implement Sequence Protocol **************************
 *************************************************************************/

/* Some of this is repeated in the array_as_mapping protocol.  But
   we fill it in here so that PySequence_XXXX calls work as expected
*/


static PyObject *
array_slice(PyArrayObject *self, Py_ssize_t ilow,
            Py_ssize_t ihigh)
{
    PyArrayObject *r;
    Py_ssize_t l;
    char *data;

    if (self->nd == 0) {
        PyErr_SetString(PyExc_ValueError, "cannot slice a 0-d array");
        return NULL;
    }

    l=self->dimensions[0];
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow > l) {
        ilow = l;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > l) {
        ihigh = l;
    }

    if (ihigh != ilow) {
        data = index2ptr(self, ilow);
        if (data == NULL) {
            return NULL;
        }
    }
    else {
        data = self->data;
    }

    self->dimensions[0] = ihigh-ilow;
    Py_INCREF(self->descr);
    r = (PyArrayObject *)                                           \
        PyArray_NewFromDescr(self->ob_type, self->descr,
                             self->nd, self->dimensions,
                             self->strides, data,
                             self->flags, (PyObject *)self);
    self->dimensions[0] = l;
    if (r == NULL) {
        return NULL;
    }
    r->base = (PyObject *)self;
    Py_INCREF(self);
    PyArray_UpdateFlags(r, UPDATE_ALL);
    return (PyObject *)r;
}


static int
array_ass_slice(PyArrayObject *self, Py_ssize_t ilow,
                Py_ssize_t ihigh, PyObject *v) {
    int ret;
    PyArrayObject *tmp;

    if (v == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (!PyArray_ISWRITEABLE(self)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array is not writeable");
        return -1;
    }
    if ((tmp = (PyArrayObject *)array_slice(self, ilow, ihigh)) == NULL) {
        return -1;
    }
    ret = PyArray_CopyObject(tmp, v);
    Py_DECREF(tmp);

    return ret;
}

static int
array_contains(PyArrayObject *self, PyObject *el)
{
    /* equivalent to (self == el).any() */

    PyObject *res;
    int ret;

    res = PyArray_EnsureAnyArray(PyObject_RichCompare((PyObject *)self,
                                                      el, Py_EQ));
    if (res == NULL) {
        return -1;
    }
    ret = array_any_nonzero((PyArrayObject *)res);
    Py_DECREF(res);
    return ret;
}

static PySequenceMethods array_as_sequence = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)array_length,                  /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,
    (ssizeargfunc)array_item_nice,
    (ssizessizeargfunc)array_slice,
    (ssizeobjargproc)array_ass_item,        /*sq_ass_item*/
    (ssizessizeobjargproc)array_ass_slice,  /*sq_ass_slice*/
    (objobjproc) array_contains,            /*sq_contains */
    (binaryfunc) NULL,                      /*sg_inplace_concat */
    (ssizeargfunc)NULL,
#else
    (inquiry)array_length,                  /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat is handled by nb_add*/
    (intargfunc)NULL,                       /*sq_repeat is handled nb_multiply*/
    (intargfunc)array_item_nice,            /*sq_item*/
    (intintargfunc)array_slice,             /*sq_slice*/
    (intobjargproc)array_ass_item,          /*sq_ass_item*/
    (intintobjargproc)array_ass_slice,      /*sq_ass_slice*/
    (objobjproc) array_contains,            /*sq_contains */
    (binaryfunc) NULL,                      /*sg_inplace_concat */
    (intargfunc) NULL                       /*sg_inplace_repeat */
#endif
};


/****************** End of Sequence Protocol ****************************/


static int
dump_data(char **string, int *n, int *max_n, char *data, int nd,
          intp *dimensions, intp *strides, PyArrayObject* self)
{
    PyArray_Descr *descr=self->descr;
    PyObject *op, *sp;
    char *ostring;
    intp i, N;

#define CHECK_MEMORY do { if (*n >= *max_n-16) {         \
        *max_n *= 2;                                     \
        *string = (char *)_pya_realloc(*string, *max_n); \
    }} while (0)

    if (nd == 0) {
        if ((op = descr->f->getitem(data, self)) == NULL) {
            return -1;
        }
        sp = PyObject_Repr(op);
        if (sp == NULL) {
            Py_DECREF(op);
            return -1;
        }
        ostring = PyString_AsString(sp);
        N = PyString_Size(sp)*sizeof(char);
        *n += N;
        CHECK_MEMORY;
        memmove(*string + (*n - N), ostring, N);
        Py_DECREF(sp);
        Py_DECREF(op);
        return 0;
    }
    else {
        CHECK_MEMORY;
        (*string)[*n] = '[';
        *n += 1;
        for (i = 0; i < dimensions[0]; i++) {
            if (dump_data(string, n, max_n,
                          data + (*strides)*i,
                          nd - 1, dimensions + 1,
                          strides + 1, self) < 0) {
                return -1;
            }
            CHECK_MEMORY;
            if (i < dimensions[0] - 1) {
                (*string)[*n] = ',';
                (*string)[*n+1] = ' ';
                *n += 2;
            }
        }
        CHECK_MEMORY;
        (*string)[*n] = ']';
        *n += 1;
        return 0;
    }

#undef CHECK_MEMORY
}

static PyObject *
array_repr_builtin(PyArrayObject *self, int repr)
{
    PyObject *ret;
    char *string;
    int n, max_n;

    max_n = PyArray_NBYTES(self)*4*sizeof(char) + 7;

    if ((string = (char *)_pya_malloc(max_n)) == NULL) {
        PyErr_SetString(PyExc_MemoryError, "out of memory");
        return NULL;
    }

    if (repr) {
        n = 6;
        sprintf(string, "array(");
    }
    else {
        n = 0;
    }
    if (dump_data(&string, &n, &max_n, self->data,
                  self->nd, self->dimensions,
                  self->strides, self) < 0) {
        _pya_free(string);
        return NULL;
    }

    if (repr) {
        if (PyArray_ISEXTENDED(self)) {
            char buf[100];
            PyOS_snprintf(buf, sizeof(buf), "%d", self->descr->elsize);
            sprintf(string+n, ", '%c%s')", self->descr->type, buf);
            ret = PyString_FromStringAndSize(string, n + 6 + strlen(buf));
        }
        else {
            sprintf(string+n, ", '%c')", self->descr->type);
            ret = PyString_FromStringAndSize(string, n+6);
        }
    }
    else {
        ret = PyString_FromStringAndSize(string, n);
    }

    _pya_free(string);
    return ret;
}

static PyObject *PyArray_StrFunction = NULL;
static PyObject *PyArray_ReprFunction = NULL;

/*NUMPY_API
 * Set the array print function to be a Python function.
 */
NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr)
{
    if (repr) {
        /* Dispose of previous callback */
        Py_XDECREF(PyArray_ReprFunction);
        /* Add a reference to new callback */
        Py_XINCREF(op);
        /* Remember new callback */
        PyArray_ReprFunction = op;
    }
    else {
        /* Dispose of previous callback */
        Py_XDECREF(PyArray_StrFunction);
        /* Add a reference to new callback */
        Py_XINCREF(op);
        /* Remember new callback */
        PyArray_StrFunction = op;
    }
}

static PyObject *
array_repr(PyArrayObject *self)
{
    PyObject *s, *arglist;

    if (PyArray_ReprFunction == NULL) {
        s = array_repr_builtin(self, 1);
    }
    else {
        arglist = Py_BuildValue("(O)", self);
        s = PyEval_CallObject(PyArray_ReprFunction, arglist);
        Py_DECREF(arglist);
    }
    return s;
}

static PyObject *
array_str(PyArrayObject *self)
{
    PyObject *s, *arglist;

    if (PyArray_StrFunction == NULL) {
        s = array_repr_builtin(self, 0);
    }
    else {
        arglist = Py_BuildValue("(O)", self);
        s = PyEval_CallObject(PyArray_StrFunction, arglist);
        Py_DECREF(arglist);
    }
    return s;
}



/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
{
    PyArray_UCS4 c1, c2;
    while(len-- > 0) {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            return (c1 < c2) ? -1 : 1;
        }
    }
    return 0;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareString(char *s1, char *s2, size_t len)
{
    const unsigned char *c1 = (unsigned char *)s1;
    const unsigned char *c2 = (unsigned char *)s2;
    size_t i;

    for(i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return (c1[i] > c2[i]) ? 1 : -1;
        }
    }
    return 0;
}


/* This also handles possibly mis-aligned data */
/* Compare s1 and s2 which are not necessarily NULL-terminated.
   s1 is of length len1
   s2 is of length len2
   If they are NULL terminated, then stop comparison.
*/
static int
_myunincmp(PyArray_UCS4 *s1, PyArray_UCS4 *s2, int len1, int len2)
{
    PyArray_UCS4 *sptr;
    PyArray_UCS4 *s1t=s1, *s2t=s2;
    int val;
    intp size;
    int diff;

    if ((intp)s1 % sizeof(PyArray_UCS4) != 0) {
        size = len1*sizeof(PyArray_UCS4);
        s1t = malloc(size);
        memcpy(s1t, s1, size);
    }
    if ((intp)s2 % sizeof(PyArray_UCS4) != 0) {
        size = len2*sizeof(PyArray_UCS4);
        s2t = malloc(size);
        memcpy(s2t, s2, size);
    }
    val = PyArray_CompareUCS4(s1t, s2t, MIN(len1,len2));
    if ((val != 0) || (len1 == len2)) {
        goto finish;
    }
    if (len2 > len1) {
        sptr = s2t+len1;
        val = -1;
        diff = len2-len1;
    }
    else {
        sptr = s1t+len2;
        val = 1;
        diff=len1-len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            goto finish;
        }
        sptr++;
    }
    val = 0;

 finish:
    if (s1t != s1) {
        free(s1t);
    }
    if (s2t != s2) {
        free(s2t);
    }
    return val;
}




/*
 * Compare s1 and s2 which are not necessarily NULL-terminated.
 * s1 is of length len1
 * s2 is of length len2
 * If they are NULL terminated, then stop comparison.
 */
static int
_mystrncmp(char *s1, char *s2, int len1, int len2)
{
    char *sptr;
    int val;
    int diff;

    val = memcmp(s1, s2, MIN(len1, len2));
    if ((val != 0) || (len1 == len2)) {
        return val;
    }
    if (len2 > len1) {
        sptr = s2 + len1;
        val = -1;
        diff = len2 - len1;
    }
    else {
        sptr = s1 + len2;
        val = 1;
        diff = len1 - len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            return val;
        }
        sptr++;
    }
    return 0; /* Only happens if NULLs are everywhere */
}

/* Borrowed from Numarray */

#define SMALL_STRING 2048

#if defined(isspace)
#undef isspace
#define isspace(c)  ((c==' ')||(c=='\t')||(c=='\n')||(c=='\r')||(c=='\v')||(c=='\f'))
#endif

static void _rstripw(char *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        int c = s[i];

        if (!c || isspace(c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}

static void _unistripw(PyArray_UCS4 *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        PyArray_UCS4 c = s[i];
        if (!c || isspace(c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}


static char *
_char_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc > SMALL_STRING) {
        temp = malloc(nc);
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc);
    _rstripw(temp, nc);
    return temp;
}

static void
_char_release(char *ptr, int nc)
{
    if (nc > SMALL_STRING) {
        free(ptr);
    }
}

static char *
_uni_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc*sizeof(PyArray_UCS4) > SMALL_STRING) {
        temp = malloc(nc*sizeof(PyArray_UCS4));
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc*sizeof(PyArray_UCS4));
    _unistripw((PyArray_UCS4 *)temp, nc);
    return temp;
}

static void
_uni_release(char *ptr, int nc)
{
    if (nc*sizeof(PyArray_UCS4) > SMALL_STRING) {
        free(ptr);
    }
}


/* End borrowed from numarray */

#define _rstrip_loop(CMP) {                                     \
        void *aptr, *bptr;                                      \
        char atemp[SMALL_STRING], btemp[SMALL_STRING];          \
        while(size--) {                                         \
            aptr = stripfunc(iself->dataptr, atemp, N1);        \
            if (!aptr) return -1;                               \
            bptr = stripfunc(iother->dataptr, btemp, N2);       \
            if (!bptr) {                                        \
                relfunc(aptr, N1);                              \
                return -1;                                      \
            }                                                   \
            val = cmpfunc(aptr, bptr, N1, N2);                  \
            *dptr = (val CMP 0);                                \
            PyArray_ITER_NEXT(iself);                           \
            PyArray_ITER_NEXT(iother);                          \
            dptr += 1;                                          \
            relfunc(aptr, N1);                                  \
            relfunc(bptr, N2);                                  \
        }                                                       \
    }

#define _reg_loop(CMP) {                                \
        while(size--) {                                 \
            val = cmpfunc((void *)iself->dataptr,       \
                          (void *)iother->dataptr,      \
                          N1, N2);                      \
            *dptr = (val CMP 0);                        \
            PyArray_ITER_NEXT(iself);                   \
            PyArray_ITER_NEXT(iother);                  \
            dptr += 1;                                  \
        }                                               \
    }

#define _loop(CMP) if (rstrip) _rstrip_loop(CMP)        \
        else _reg_loop(CMP)

static int
_compare_strings(PyObject *result, PyArrayMultiIterObject *multi,
                 int cmp_op, void *func, int rstrip)
{
    PyArrayIterObject *iself, *iother;
    Bool *dptr;
    intp size;
    int val;
    int N1, N2;
    int (*cmpfunc)(void *, void *, int, int);
    void (*relfunc)(char *, int);
    char* (*stripfunc)(char *, char *, int);

    cmpfunc = func;
    dptr = (Bool *)PyArray_DATA(result);
    iself = multi->iters[0];
    iother = multi->iters[1];
    size = multi->size;
    N1 = iself->ao->descr->elsize;
    N2 = iother->ao->descr->elsize;
    if ((void *)cmpfunc == (void *)_myunincmp) {
        N1 >>= 2;
        N2 >>= 2;
        stripfunc = _uni_copy_n_strip;
        relfunc = _uni_release;
    }
    else {
        stripfunc = _char_copy_n_strip;
        relfunc = _char_release;
    }
    switch (cmp_op) {
    case Py_EQ:
        _loop(==)
            break;
    case Py_NE:
        _loop(!=)
            break;
    case Py_LT:
        _loop(<)
            break;
    case Py_LE:
        _loop(<=)
            break;
    case Py_GT:
        _loop(>)
            break;
    case Py_GE:
        _loop(>=)
            break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "bad comparison operator");
        return -1;
    }
    return 0;
}

#undef _loop
#undef _reg_loop
#undef _rstrip_loop
#undef SMALL_STRING

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip)
{
    PyObject *result;
    PyArrayMultiIterObject *mit;
    int val;

    /* Cast arrays to a common type */
    if (self->descr->type_num != other->descr->type_num) {
        PyObject *new;
        if (self->descr->type_num == PyArray_STRING &&
            other->descr->type_num == PyArray_UNICODE) {
            Py_INCREF(other->descr);
            new = PyArray_FromAny((PyObject *)self, other->descr,
                                  0, 0, 0, NULL);
            if (new == NULL) {
                return NULL;
            }
            Py_INCREF(other);
            self = (PyArrayObject *)new;
        }
        else if (self->descr->type_num == PyArray_UNICODE &&
                 other->descr->type_num == PyArray_STRING) {
            Py_INCREF(self->descr);
            new = PyArray_FromAny((PyObject *)other, self->descr,
                                  0, 0, 0, NULL);
            if (new == NULL) {
                return NULL;
            }
            Py_INCREF(self);
            other = (PyArrayObject *)new;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "invalid string data-types "
                            "in comparison");
            return NULL;
        }
    }
    else {
        Py_INCREF(self);
        Py_INCREF(other);
    }

    /* Broad-cast the arrays to a common shape */
    mit = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, self, other);
    Py_DECREF(self);
    Py_DECREF(other);
    if (mit == NULL) {
        return NULL;
    }

    result = PyArray_NewFromDescr(&PyArray_Type,
                                  PyArray_DescrFromType(PyArray_BOOL),
                                  mit->nd,
                                  mit->dimensions,
                                  NULL, NULL, 0,
                                  NULL);
    if (result == NULL) {
        goto finish;
    }

    if (self->descr->type_num == PyArray_UNICODE) {
        val = _compare_strings(result, mit, cmp_op, _myunincmp, rstrip);
    }
    else {
        val = _compare_strings(result, mit, cmp_op, _mystrncmp, rstrip);
    }

    if (val < 0) {
        Py_DECREF(result); result = NULL;
    }

 finish:
    Py_DECREF(mit);
    return result;
}

/*
 * VOID-type arrays can only be compared equal and not-equal
 * in which case the fields are all compared by extracting the fields
 * and testing one at a time...
 * equality testing is performed using logical_ands on all the fields.
 * in-equality testing is performed using logical_ors on all the fields.
 *
 * VOID-type arrays without fields are compared for equality by comparing their
 * memory at each location directly (using string-code).
 */
static PyObject *
_void_compare(PyArrayObject *self, PyArrayObject *other, int cmp_op)
{
    if (!(cmp_op == Py_EQ || cmp_op == Py_NE)) {
        PyErr_SetString(PyExc_ValueError,
                "Void-arrays can only be compared for equality.");
        return NULL;
    }
    if (PyArray_HASFIELDS(self)) {
        PyObject *res = NULL, *temp, *a, *b;
        PyObject *key, *value, *temp2;
        PyObject *op;
        Py_ssize_t pos = 0;

        op = (cmp_op == Py_EQ ? n_ops.logical_and : n_ops.logical_or);
        while (PyDict_Next(self->descr->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            a = PyArray_EnsureAnyArray(array_subscript(self, key));
            if (a == NULL) {
                Py_XDECREF(res);
                return NULL;
            }
            b = array_subscript(other, key);
            if (b == NULL) {
                Py_XDECREF(res);
                Py_DECREF(a);
                return NULL;
            }
            temp = array_richcompare((PyArrayObject *)a,b,cmp_op);
            Py_DECREF(a);
            Py_DECREF(b);
            if (temp == NULL) {
                Py_XDECREF(res);
                return NULL;
            }
            if (res == NULL) {
                res = temp;
            }
            else {
                temp2 = PyObject_CallFunction(op, "OO", res, temp);
                Py_DECREF(temp);
                Py_DECREF(res);
                if (temp2 == NULL) {
                    return NULL;
                }
                res = temp2;
            }
        }
        if (res == NULL && !PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "No fields found.");
        }
        return res;
    }
    else {
        /*
         * compare as a string. Assumes self and
         * other have same descr->type
         */
        return _strings_richcompare(self, other, cmp_op, 0);
    }
}

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
{
    PyObject *array_other, *result = NULL;
    int typenum;

    switch (cmp_op) {
    case Py_LT:
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.less);
        break;
    case Py_LE:
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.less_equal);
        break;
    case Py_EQ:
        if (other == Py_None) {
            Py_INCREF(Py_False);
            return Py_False;
        }
        /* Try to convert other to an array */
        if (!PyArray_Check(other)) {
            typenum = self->descr->type_num;
            if (typenum != PyArray_OBJECT) {
                typenum = PyArray_NOTYPE;
            }
            array_other = PyArray_FromObject(other,
                    typenum, 0, 0);
            /*
             * If not successful, then return False. This fixes code
             * that used to allow equality comparisons between arrays
             * and other objects which would give a result of False.
             */
            if ((array_other == NULL) ||
                    (array_other == Py_None)) {
                Py_XDECREF(array_other);
                PyErr_Clear();
                Py_INCREF(Py_False);
                return Py_False;
            }
        }
        else {
            Py_INCREF(other);
            array_other = other;
        }
        result = PyArray_GenericBinaryFunction(self,
                array_other,
                n_ops.equal);
        if ((result == Py_NotImplemented) &&
                (self->descr->type_num == PyArray_VOID)) {
            int _res;

            _res = PyObject_RichCompareBool
                ((PyObject *)self->descr,
                 (PyObject *)\
                 PyArray_DESCR(array_other),
                 Py_EQ);
            if (_res < 0) {
                Py_DECREF(result);
                Py_DECREF(array_other);
                return NULL;
            }
            if (_res) {
                Py_DECREF(result);
                result = _void_compare
                    (self,
                     (PyArrayObject *)array_other,
                     cmp_op);
                Py_DECREF(array_other);
            }
            return result;
        }
        /*
         * If the comparison results in NULL, then the
         * two array objects can not be compared together so
         * return zero
         */
        Py_DECREF(array_other);
        if (result == NULL) {
            PyErr_Clear();
            Py_INCREF(Py_False);
            return Py_False;
        }
        break;
    case Py_NE:
        if (other == Py_None) {
            Py_INCREF(Py_True);
            return Py_True;
        }
        /* Try to convert other to an array */
        if (!PyArray_Check(other)) {
            typenum = self->descr->type_num;
            if (typenum != PyArray_OBJECT) {
                typenum = PyArray_NOTYPE;
            }
            array_other = PyArray_FromObject(other, typenum, 0, 0);
            /*
             * If not successful, then objects cannot be
             * compared and cannot be equal, therefore,
             * return True;
             */
            if ((array_other == NULL) || (array_other == Py_None)) {
                Py_XDECREF(array_other);
                PyErr_Clear();
                Py_INCREF(Py_True);
                return Py_True;
            }
        }
        else {
            Py_INCREF(other);
            array_other = other;
        }
        result = PyArray_GenericBinaryFunction(self,
                array_other,
                n_ops.not_equal);
        if ((result == Py_NotImplemented) &&
                (self->descr->type_num == PyArray_VOID)) {
            int _res;

            _res = PyObject_RichCompareBool(
                    (PyObject *)self->descr,
                    (PyObject *)
                    PyArray_DESCR(array_other),
                    Py_EQ);
            if (_res < 0) {
                Py_DECREF(result);
                Py_DECREF(array_other);
                return NULL;
            }
            if (_res) {
                Py_DECREF(result);
                result = _void_compare(
                        self,
                        (PyArrayObject *)array_other,
                        cmp_op);
                Py_DECREF(array_other);
            }
            return result;
        }

        Py_DECREF(array_other);
        if (result == NULL) {
            PyErr_Clear();
            Py_INCREF(Py_True);
            return Py_True;
        }
        break;
    case Py_GT:
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater);
        break;
    case Py_GE:
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater_equal);
        break;
    default:
        result = Py_NotImplemented;
        Py_INCREF(result);
    }
    if (result == Py_NotImplemented) {
        /* Try to handle string comparisons */
        if (self->descr->type_num == PyArray_OBJECT) {
            return result;
        }
        array_other = PyArray_FromObject(other,PyArray_NOTYPE, 0, 0);
        if (PyArray_ISSTRING(self) && PyArray_ISSTRING(array_other)) {
            Py_DECREF(result);
            result = _strings_richcompare(self, (PyArrayObject *)
                                          array_other, cmp_op, 0);
        }
        Py_DECREF(array_other);
    }
    return result;
}


/*NUMPY_API
  PyArray_CheckAxis
*/
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    PyObject *temp1, *temp2;
    int n = arr->nd;

    if ((*axis >= MAX_DIMS) || (n==0)) {
        if (n != 1) {
            temp1 = PyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            *axis = PyArray_NDIM(temp1)-1;
        }
        else {
            temp1 = (PyObject *)arr;
            Py_INCREF(temp1);
            *axis = 0;
        }
        if (!flags) {
            return temp1;
        }
    }
    else {
        temp1 = (PyObject *)arr;
        Py_INCREF(temp1);
    }
    if (flags) {
        temp2 = PyArray_CheckFromAny((PyObject *)temp1, NULL,
                                     0, 0, flags, NULL);
        Py_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    }
    else {
        temp2 = (PyObject *)temp1;
    }
    n = PyArray_NDIM(temp2);
    if (*axis < 0) {
        *axis += n;
    }
    if ((*axis < 0) || (*axis >= n)) {
        PyErr_Format(PyExc_ValueError,
                     "axis(=%d) out of bounds", *axis);
        Py_DECREF(temp2);
        return NULL;
    }
    return temp2;
}

#define _check_axis PyArray_CheckAxis

#include "arraymethods.c"

/* Lifted from numarray */
/*NUMPY_API
  PyArray_IntTupleFromIntp
*/
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, intp *vals)
{
    int i;
    PyObject *intTuple = PyTuple_New(len);

    if (!intTuple) {
        goto fail;
    }
    for (i = 0; i < len; i++) {
#if SIZEOF_INTP <= SIZEOF_LONG
        PyObject *o = PyInt_FromLong((long) vals[i]);
#else
        PyObject *o = PyLong_FromLongLong((longlong) vals[i]);
#endif
        if (!o) {
            Py_DECREF(intTuple);
            intTuple = NULL;
            goto fail;
        }
        PyTuple_SET_ITEM(intTuple, i, o);
    }

 fail:
    return intTuple;
}

/*NUMPY_API
 * PyArray_IntpFromSequence
 * Returns the number of dimensions or -1 if an error occurred.
 * vals must be large enough to hold maxvals
 */
NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, intp *vals, int maxvals)
{
    int nd, i;
    PyObject *op, *err;

    /*
     * Check to see if sequence is a single integer first.
     * or, can be made into one
     */
    if ((nd=PySequence_Length(seq)) == -1) {
        if (PyErr_Occurred()) PyErr_Clear();
#if SIZEOF_LONG >= SIZEOF_INTP
        if (!(op = PyNumber_Int(seq))) {
            return -1;
        }
#else
        if (!(op = PyNumber_Long(seq))) {
            return -1;
        }
#endif
        nd = 1;
#if SIZEOF_LONG >= SIZEOF_INTP
        vals[0] = (intp ) PyInt_AsLong(op);
#else
        vals[0] = (intp ) PyLong_AsLongLong(op);
#endif
        Py_DECREF(op);

        /*
         * Check wether there was an error - if the error was an overflow, raise
         * a ValueError instead to be more helpful
         */
        if(vals[0] == -1) {
            err = PyErr_Occurred();
            if (err  &&
                PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                PyErr_SetString(PyExc_ValueError,
                        "Maximum allowed dimension exceeded");
            }
            if(err != NULL) {
                return -1;
            }
        }
    }
    else {
        for (i = 0; i < MIN(nd,maxvals); i++) {
            op = PySequence_GetItem(seq, i);
            if (op == NULL) {
                return -1;
            }
#if SIZEOF_LONG >= SIZEOF_INTP
            vals[i]=(intp )PyInt_AsLong(op);
#else
            vals[i]=(intp )PyLong_AsLongLong(op);
#endif
            Py_DECREF(op);

            /*
             * Check wether there was an error - if the error was an overflow,
             * raise a ValueError instead to be more helpful
             */
            if(vals[0] == -1) {
                err = PyErr_Occurred();
                if (err  &&
                    PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                    PyErr_SetString(PyExc_ValueError,
                            "Maximum allowed dimension exceeded");
                }
                if(err != NULL) {
                    return -1;
                }
            }
        }
    }
    return nd;
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

static int
_IsAligned(PyArrayObject *ap)
{
    int i, alignment, aligned = 1;
    intp ptr;
    int type = ap->descr->type_num;

    if ((type == PyArray_STRING) || (type == PyArray_VOID)) {
        return 1;
    }
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

static Bool
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


/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_ElementStrides(PyObject *arr)
{
    int itemsize = PyArray_ITEMSIZE(arr);
    int i, N = PyArray_NDIM(arr);
    intp *strides = PyArray_STRIDES(arr);

    for (i = 0; i < N; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
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
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */

/*NUMPY_API*/
NPY_NO_EXPORT Bool
PyArray_CheckStrides(int elsize, int nd, intp numbytes, intp offset,
                     intp *dims, intp *newstrides)
{
    int i;
    intp byte_begin;
    intp begin;
    intp end;

    if (numbytes == 0) {
        numbytes = PyArray_MultiplyList(dims, nd) * elsize;
    }
    begin = -offset;
    end = numbytes - offset - elsize;
    for (i = 0; i < nd; i++) {
        byte_begin = newstrides[i]*(dims[i] - 1);
        if ((byte_begin < begin) || (byte_begin > end)) {
            return FALSE;
        }
    }
    return TRUE;
}


/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the CONTIGUOUS bit will be set.  If the flags argument
 * has the FORTRAN bit set, then a FORTRAN-style strides array will be
 * created (and of course the FORTRAN flag bit will be set).
 *
 * If data is not given but created here, then flags will be DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 */

static size_t
_array_fill_strides(intp *strides, intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & FORTRAN) && !(inflag & CONTIGUOUS)) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        *objflags |= FORTRAN;
        if (nd > 1) {
            *objflags &= ~CONTIGUOUS;
        }
        else {
            *objflags |= CONTIGUOUS;
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        *objflags |= CONTIGUOUS;
        if (nd > 1) {
            *objflags &= ~FORTRAN;
        }
        else {
            *objflags |= FORTRAN;
        }
    }
    return itemsize;
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(PyTypeObject *subtype, int nd, intp *dims, int type_num,
            intp *strides, void *data, int itemsize, int flags,
            PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (descr->elsize == 0) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        descr->elsize = itemsize;
    }
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}

/*
 * Change a sub-array field to the base descriptor
 *
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end unless we have a FORTRAN array
 * and then they are added to the beginning
 *
 * Strides are only added if given (because data is given).
 */
static int
_update_descr_and_dimensions(PyArray_Descr **des, intp *newdims,
                             intp *newstrides, int oldnd, int isfortran)
{
    PyArray_Descr *old;
    int newnd;
    int numnew;
    intp *mydim;
    int i;
    int tuple;

    old = *des;
    *des = old->subarray->base;


    mydim = newdims + oldnd;
    tuple = PyTuple_Check(old->subarray->shape);
    if (tuple) {
        numnew = PyTuple_GET_SIZE(old->subarray->shape);
    }
    else {
        numnew = 1;
    }


    newnd = oldnd + numnew;
    if (newnd > MAX_DIMS) {
        goto finish;
    }
    if (isfortran) {
        memmove(newdims+numnew, newdims, oldnd*sizeof(intp));
        mydim = newdims;
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (intp) PyInt_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));
        }
    }
    else {
        mydim[0] = (intp) PyInt_AsLong(old->subarray->shape);
    }

    if (newstrides) {
        intp tempsize;
        intp *mystrides;

        mystrides = newstrides + oldnd;
        if (isfortran) {
            memmove(newstrides+numnew, newstrides, oldnd*sizeof(intp));
            mystrides = newstrides;
        }
        /* Make new strides -- alwasy C-contiguous */
        tempsize = (*des)->elsize;
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;
            tempsize *= mydim[i] ? mydim[i] : 1;
        }
    }

 finish:
    Py_INCREF(*des);
    Py_DECREF(old);
    return newnd;
}


/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr (even on failure)
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     intp *dims, intp *strides, void *data,
                     int flags, PyObject *obj)
{
    PyArrayObject *self;
    int i;
    size_t sd;
    intp largest;
    intp size;

    if (descr->subarray) {
        PyObject *ret;
        intp newdims[2*MAX_DIMS];
        intp *newstrides = NULL;
        int isfortran = 0;
        isfortran = (data && (flags & FORTRAN) && !(flags & CONTIGUOUS)) ||
            (!data && flags);
        memcpy(newdims, dims, nd*sizeof(intp));
        if (strides) {
            newstrides = newdims + MAX_DIMS;
            memcpy(newstrides, strides, nd*sizeof(intp));
        }
        nd =_update_descr_and_dimensions(&descr, newdims,
                                         newstrides, nd, isfortran);
        ret = PyArray_NewFromDescr(subtype, descr, nd, newdims,
                                   newstrides,
                                   data, flags, obj);
        return ret;
    }
    if (nd < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "number of dimensions must be >=0");
        Py_DECREF(descr);
        return NULL;
    }
    if (nd > MAX_DIMS) {
        PyErr_Format(PyExc_ValueError,
                     "maximum number of dimensions is %d", MAX_DIMS);
        Py_DECREF(descr);
        return NULL;
    }

    /* Check dimensions */
    size = 1;
    sd = (size_t) descr->elsize;
    if (sd == 0) {
        if (!PyDataType_ISSTRING(descr)) {
            PyErr_SetString(PyExc_ValueError, "Empty data-type");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        if (descr->type_num == NPY_STRING) {
            descr->elsize = 1;
        }
        else {
            descr->elsize = sizeof(PyArray_UCS4);
        }
        sd = descr->elsize;
    }

    largest = NPY_MAX_INTP / sd;
    for (i = 0; i < nd; i++) {
        intp dim = dims[i];

        if (dim == 0) {
            /*
             * Compare to PyArray_OverflowMultiplyList that
             * returns 0 in this case.
             */
            continue;
        }
        if (dim < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "negative dimensions "  \
                            "are not allowed");
            Py_DECREF(descr);
            return NULL;
        }
        if (dim > largest) {
            PyErr_SetString(PyExc_ValueError,
                            "array is too big.");
            Py_DECREF(descr);
            return NULL;
        }
        size *= dim;
        largest /= dim;
    }

    self = (PyArrayObject *) subtype->tp_alloc(subtype, 0);
    if (self == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    self->nd = nd;
    self->dimensions = NULL;
    self->data = NULL;
    if (data == NULL) {
        self->flags = DEFAULT;
        if (flags) {
            self->flags |= FORTRAN;
            if (nd > 1) {
                self->flags &= ~CONTIGUOUS;
            }
            flags = FORTRAN;
        }
    }
    else {
        self->flags = (flags & ~UPDATEIFCOPY);
    }
    self->descr = descr;
    self->base = (PyObject *)NULL;
    self->weakreflist = (PyObject *)NULL;

    if (nd > 0) {
        self->dimensions = PyDimMem_NEW(2*nd);
        if (self->dimensions == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        self->strides = self->dimensions + nd;
        memcpy(self->dimensions, dims, sizeof(intp)*nd);
        if (strides == NULL) { /* fill it in */
            sd = _array_fill_strides(self->strides, dims, nd, sd,
                                     flags, &(self->flags));
        }
        else {
            /*
             * we allow strides even when we create
             * the memory, but be careful with this...
             */
            memcpy(self->strides, strides, sizeof(intp)*nd);
            sd *= size;
        }
    }
    else {
        self->dimensions = self->strides = NULL;
    }

    if (data == NULL) {
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         */

        if (sd == 0) {
            sd = descr->elsize;
        }
        if ((data = PyDataMem_NEW(sd)) == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        self->flags |= OWNDATA;

        /*
         * It is bad to have unitialized OBJECT pointers
         * which could also be sub-fields of a VOID array
         */
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(data, 0, sd);
        }
    }
    else {
        /*
         * If data is passed in, this object won't own it by default.
         * Caller must arrange for this to be reset if truly desired
         */
        self->flags &= ~OWNDATA;
    }
    self->data = data;

    /*
     * call the __array_finalize__
     * method if a subtype.
     * If obj is NULL, then call method with Py_None
     */
    if ((subtype != &PyArray_Type)) {
        PyObject *res, *func, *args;
        static PyObject *str = NULL;

        if (str == NULL) {
            str = PyString_InternFromString("__array_finalize__");
        }
        func = PyObject_GetAttr((PyObject *)self, str);
        if (func && func != Py_None) {
            if (strides != NULL) {
                /*
                 * did not allocate own data or funny strides
                 * update flags before finalize function
                 */
                PyArray_UpdateFlags(self, UPDATE_ALL);
            }
            if PyCObject_Check(func) {
                /* A C-function is stored here */
                PyArray_FinalizeFunc *cfunc;
                cfunc = PyCObject_AsVoidPtr(func);
                Py_DECREF(func);
                if (cfunc(self, obj) < 0) {
                    goto fail;
                }
            }
            else {
                args = PyTuple_New(1);
                if (obj == NULL) {
                    obj=Py_None;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(args, 0, obj);
                res = PyObject_Call(func, args, NULL);
                Py_DECREF(args);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
        else Py_XDECREF(func);
    }
    return (PyObject *)self;

 fail:
    Py_DECREF(self);
    return NULL;
}

static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        memset(optr, 0, dtype->elsize);
    }
    else if (PyDescr_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _putzero(optr + offset, zero, new);
        }
    }
    else {
        PyObject **temp;
        Py_INCREF(zero);
        temp = (PyObject **)optr;
        *temp = zero;
    }
    return;
}


/*NUMPY_API
 * Resize (reallocate data).  Only works if nothing else is referencing this
 * array and it is contiguous.  If refcheck is 0, then the reference count is
 * not checked and assumed to be 1.  You still must own this data and have no
 * weak-references and no base object.
 */
NPY_NO_EXPORT PyObject *
PyArray_Resize(PyArrayObject *self, PyArray_Dims *newshape, int refcheck,
               NPY_ORDER fortran)
{
    intp oldsize, newsize;
    int new_nd=newshape->len, k, n, elsize;
    int refcnt;
    intp* new_dimensions=newshape->ptr;
    intp new_strides[MAX_DIMS];
    size_t sd;
    intp *dimptr;
    char *new_data;
    intp largest;

    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                        "resize only works on single-segment arrays");
        return NULL;
    }

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_CORDER;
    }
    if (self->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "Bad data-type size.");
        return NULL;
    }
    newsize = 1;
    largest = MAX_INTP / self->descr->elsize;
    for(k=0; k<new_nd; k++) {
        if (new_dimensions[k]==0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "negative dimensions not allowed");
            return NULL;
        }
        newsize *= new_dimensions[k];
        if (newsize <=0 || newsize > largest) {
            return PyErr_NoMemory();
        }
    }
    oldsize = PyArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(self->flags & OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot resize this array:  "   \
                            "it does not own its data");
            return NULL;
        }

        if (refcheck) {
            refcnt = REFCOUNT(self);
        }
        else {
            refcnt = 1;
        }
        if ((refcnt > 2) || (self->base != NULL) ||
            (self->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot resize an array that has "\
                            "been referenced or is referencing\n"\
                            "another array in this way.  Use the "\
                            "resize function");
            return NULL;
        }

        if (newsize == 0) {
            sd = self->descr->elsize;
        }
        else {
            sd = newsize*self->descr->elsize;
        }
        /* Reallocate space if needed */
        new_data = PyDataMem_RENEW(self->data, sd);
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                            "cannot allocate memory for array");
            return NULL;
        }
        self->data = new_data;
    }

    if ((newsize > oldsize) && PyArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = self->descr->elsize;
        if (PyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyInt_FromLong(0);
            char *optr;
            optr = self->data + oldsize*elsize;
            n = newsize - oldsize;
            for (k = 0; k < n; k++) {
                _putzero((char *)optr, zero, self->descr);
                optr += elsize;
            }
            Py_DECREF(zero);
        }
        else{
            memset(self->data+oldsize*elsize, 0, (newsize-oldsize)*elsize);
        }
    }

    if (self->nd != new_nd) {
        /* Different number of dimensions. */
        self->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = PyDimMem_RENEW(self->dimensions, 2*new_nd);
        if (dimptr == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                            "cannot allocate memory for array " \
                            "(array may be corrupted)");
            return NULL;
        }
        self->dimensions = dimptr;
        self->strides = dimptr + new_nd;
    }

    /* make new_strides variable */
    sd = (size_t) self->descr->elsize;
    sd = (size_t) _array_fill_strides(new_strides, new_dimensions, new_nd, sd,
                                      self->flags, &(self->flags));
    memmove(self->dimensions, new_dimensions, new_nd*sizeof(intp));
    memmove(self->strides, new_strides, new_nd*sizeof(intp));
    Py_INCREF(Py_None);
    return Py_None;
}

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        if ((obj == Py_None) || (PyInt_Check(obj) && PyInt_AsLong(obj)==0)) {
            return;
        }
        else {
            PyObject *arr;
            Py_INCREF(dtype);
            arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                       0, NULL, NULL, NULL,
                                       0, NULL);
            if (arr!=NULL) {
                dtype->f->setitem(obj, optr, arr);
            }
            Py_XDECREF(arr);
        }
    }
    else if (PyDescr_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _fillobject(optr + offset, obj, new);
        }
    }
    else {
        PyObject **temp;
        Py_XINCREF(obj);
        temp = (PyObject **)optr;
        *temp = obj;
        return;
    }
}

/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    intp i,n;
    n = PyArray_SIZE(arr);
    if (arr->descr->type_num == PyArray_OBJECT) {
        PyObject **optr;
        optr = (PyObject **)(arr->data);
        n = PyArray_SIZE(arr);
        if (obj == NULL) {
            for (i = 0; i < n; i++) {
                *optr++ = NULL;
            }
        }
        else {
            for (i = 0; i < n; i++) {
                Py_INCREF(obj);
                *optr++ = obj;
            }
        }
    }
    else {
        char *optr;
        optr = arr->data;
        for (i = 0; i < n; i++) {
            _fillobject(optr, obj, arr->descr);
            optr += arr->descr->elsize;
        }
    }
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_FillWithScalar(PyArrayObject *arr, PyObject *obj)
{
    PyObject *newarr;
    int itemsize, swap;
    void *fromptr;
    PyArray_Descr *descr;
    intp size;
    PyArray_CopySwapFunc *copyswap;

    itemsize = arr->descr->elsize;
    if (PyArray_ISOBJECT(arr)) {
        fromptr = &obj;
        swap = 0;
        newarr = NULL;
    }
    else {
        descr = PyArray_DESCR(arr);
        Py_INCREF(descr);
        newarr = PyArray_FromAny(obj, descr, 0,0, ALIGNED, NULL);
        if (newarr == NULL) {
            return -1;
        }
        fromptr = PyArray_DATA(newarr);
        swap = (PyArray_ISNOTSWAPPED(arr) != PyArray_ISNOTSWAPPED(newarr));
    }
    size=PyArray_SIZE(arr);
    copyswap = arr->descr->f->copyswap;
    if (PyArray_ISONESEGMENT(arr)) {
        char *toptr=PyArray_DATA(arr);
        PyArray_FillWithScalarFunc* fillwithscalar =
            arr->descr->f->fillwithscalar;
        if (fillwithscalar && PyArray_ISALIGNED(arr)) {
            copyswap(fromptr, NULL, swap, newarr);
            fillwithscalar(toptr, size, fromptr, arr);
        }
        else {
            while (size--) {
                copyswap(toptr, fromptr, swap, arr);
                toptr += itemsize;
            }
        }
    }
    else {
        PyArrayIterObject *iter;

        iter = (PyArrayIterObject *)\
            PyArray_IterNew((PyObject *)arr);
        if (iter == NULL) {
            Py_XDECREF(newarr);
            return -1;
        }
        while (size--) {
            copyswap(iter->dataptr, fromptr, swap, arr);
            PyArray_ITER_NEXT(iter);
        }
        Py_DECREF(iter);
    }
    Py_XDECREF(newarr);
    return 0;
}

static PyObject *
array_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape", "dtype", "buffer",
                             "offset", "strides",
                             "order", NULL};
    PyArray_Descr *descr=NULL;
    int itemsize;
    PyArray_Dims dims = {NULL, 0};
    PyArray_Dims strides = {NULL, 0};
    PyArray_Chunk buffer;
    longlong offset=0;
    NPY_ORDER order=PyArray_CORDER;
    int fortran = 0;
    PyArrayObject *ret;

    buffer.ptr = NULL;
    /*
     * Usually called with shape and type but can also be called with buffer,
     * strides, and swapped info For now, let's just use this to create an
     * empty, contiguous array of a specific type and shape.
     */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&LO&O&",
                                     kwlist, PyArray_IntpConverter,
                                     &dims,
                                     PyArray_DescrConverter,
                                     &descr,
                                     PyArray_BufferConverter,
                                     &buffer,
                                     &offset,
                                     &PyArray_IntpConverter,
                                     &strides,
                                     &PyArray_OrderConverter,
                                     &order)) {
        goto fail;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = 1;
    }
    if (descr == NULL) {
        descr = PyArray_DescrFromType(PyArray_DEFAULT);
    }

    itemsize = descr->elsize;
    if (itemsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "data-type with unspecified variable length");
        goto fail;
    }

    if (strides.ptr != NULL) {
        intp nb, off;
        if (strides.len != dims.len) {
            PyErr_SetString(PyExc_ValueError,
                            "strides, if given, must be "   \
                            "the same length as shape");
            goto fail;
        }

        if (buffer.ptr == NULL) {
            nb = 0;
            off = 0;
        }
        else {
            nb = buffer.len;
            off = (intp) offset;
        }


        if (!PyArray_CheckStrides(itemsize, dims.len,
                                  nb, off,
                                  dims.ptr, strides.ptr)) {
            PyErr_SetString(PyExc_ValueError,
                            "strides is incompatible "      \
                            "with shape of requested "      \
                            "array and size of buffer");
            goto fail;
        }
    }

    if (buffer.ptr == NULL) {
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(subtype, descr,
                                 (int)dims.len,
                                 dims.ptr,
                                 strides.ptr, NULL, fortran, NULL);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
            /* place Py_None in object positions */
            PyArray_FillObjectArray(ret, Py_None);
            if (PyErr_Occurred()) {
                descr = NULL;
                goto fail;
            }
        }
    }
    else {
        /* buffer given -- use it */
        if (dims.len == 1 && dims.ptr[0] == -1) {
            dims.ptr[0] = (buffer.len-(intp)offset) / itemsize;
        }
        else if ((strides.ptr == NULL) &&
                 (buffer.len < ((intp)itemsize)*
                  PyArray_MultiplyList(dims.ptr, dims.len))) {
            PyErr_SetString(PyExc_TypeError,
                            "buffer is too small for "      \
                            "requested array");
            goto fail;
        }
        /* get writeable and aligned */
        if (fortran) {
            buffer.flags |= FORTRAN;
        }
        ret = (PyArrayObject *)\
            PyArray_NewFromDescr(subtype, descr,
                                 dims.len, dims.ptr,
                                 strides.ptr,
                                 offset + (char *)buffer.ptr,
                                 buffer.flags, NULL);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        PyArray_UpdateFlags(ret, UPDATE_ALL);
        ret->base = buffer.base;
        Py_INCREF(buffer.base);
    }

    PyDimMem_FREE(dims.ptr);
    if (strides.ptr) {
        PyDimMem_FREE(strides.ptr);
    }
    return (PyObject *)ret;

 fail:
    Py_XDECREF(descr);
    if (dims.ptr) {
        PyDimMem_FREE(dims.ptr);
    }
    if (strides.ptr) {
        PyDimMem_FREE(strides.ptr);
    }
    return NULL;
}


static PyObject *
array_iter(PyArrayObject *arr)
{
    if (arr->nd == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "iteration over a 0-d array");
        return NULL;
    }
    return PySeqIter_New((PyObject *)arr);
}


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
    intp nbytes;
    if (!(PyArray_ISONESEGMENT(self))) {
        PyErr_SetString(PyExc_AttributeError, "cannot get single-"\
                        "segment buffer for discontiguous array");
        return NULL;
    }
    nbytes = PyArray_NBYTES(self);
    if PyArray_ISWRITEABLE(self) {
        return PyBuffer_FromReadWriteObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
    else {
        return PyBuffer_FromObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
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
    return PyCObject_FromVoidPtrAndDesc(inter, self, gentype_struct_free);
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
        PyArray_NewFromDescr(self->ob_type,
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

static PyObject *
array_imag_get(PyArrayObject *self)
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 1);
    }
    else {
        Py_INCREF(self->descr);
	ret = (PyArrayObject *)PyArray_NewFromDescr(self->ob_type,
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

static PyGetSetDef array_getsetlist[] = {
    {"ndim",
     (getter)array_ndim_get,
     NULL, NULL, NULL},
    {"flags",
     (getter)array_flags_get,
     NULL, NULL, NULL},
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
     NULL, NULL, NULL},
    {"size",
     (getter)array_size_get,
     NULL, NULL, NULL},
    {"nbytes",
     (getter)array_nbytes_get,
     NULL, NULL, NULL},
    {"base",
     (getter)array_base_get,
     NULL, NULL, NULL},
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
     NULL, NULL, NULL},
    {"T",
     (getter)array_transpose_get,
     NULL, NULL, NULL},
    {"__array_interface__",
     (getter)array_interface_get,
     NULL, NULL, NULL},
    {"__array_struct__",
     (getter)array_struct_get,
     NULL, NULL, NULL},
    {"__array_priority__",
     (getter)array_priority_get,
     NULL, NULL, NULL},
    {"__array_finalize__",
     (getter)array_finalize_get,
     NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};

/****************** end of attribute get and set routines *******************/


static PyObject *
array_alloc(PyTypeObject *type, Py_ssize_t NPY_UNUSED(nitems))
{
    PyObject *obj;
    /* nitems will always be 0 */
    obj = (PyObject *)_pya_malloc(sizeof(PyArrayObject));
    PyObject_Init(obj, type);
    return obj;
}


NPY_NO_EXPORT PyTypeObject PyArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
    "numpy.ndarray",                             /* tp_name */
    sizeof(PyArrayObject),                       /* tp_basicsize */
    0,                                           /* tp_itemsize */
    /* methods */
    (destructor)array_dealloc,                   /* tp_dealloc */
    (printfunc)NULL,                             /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    (cmpfunc)0,                                  /* tp_compare */
    (reprfunc)array_repr,                        /* tp_repr */
    &array_as_number,                            /* tp_as_number */
    &array_as_sequence,                          /* tp_as_sequence */
    &array_as_mapping,                           /* tp_as_mapping */
    (hashfunc)0,                                 /* tp_hash */
    (ternaryfunc)0,                              /* tp_call */
    (reprfunc)array_str,                         /* tp_str */
    (getattrofunc)0,                             /* tp_getattro */
    (setattrofunc)0,                             /* tp_setattro */
    &array_as_buffer,                            /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
     | Py_TPFLAGS_BASETYPE
     | Py_TPFLAGS_CHECKTYPES),                   /* tp_flags */
    /*Documentation string */
    0,                                           /* tp_doc */

    (traverseproc)0,                             /* tp_traverse */
    (inquiry)0,                                  /* tp_clear */
    (richcmpfunc)array_richcompare,              /* tp_richcompare */
    offsetof(PyArrayObject, weakreflist),        /* tp_weaklistoffset */

    /* Iterator support (use standard) */

    (getiterfunc)array_iter,                     /* tp_iter */
    (iternextfunc)0,                             /* tp_iternext */

    /* Sub-classing (new-style object) support */

    array_methods,                               /* tp_methods */
    0,                                           /* tp_members */
    array_getsetlist,                            /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)0,                                 /* tp_init */
    array_alloc,                                 /* tp_alloc */
    (newfunc)array_new,                          /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */

#ifdef COUNT_ALLOCS
    /* these must be last and never explicitly initialized */
    0,                                           /* tp_allocs */
    0,                                           /* tp_frees */
    0,                                           /* tp_maxalloc */
    0,                                           /* tp_prev */
    0,                                           /* *tp_next */
#endif
};

/*
 * The rest of this code is to build the right kind of array
 * from a python object.
 */

static int
discover_depth(PyObject *s, int max, int stop_at_string, int stop_at_tuple)
{
    int d = 0;
    PyObject *e;

    if(max < 1) {
        return -1;
    }
    if(!PySequence_Check(s) || PyInstance_Check(s) ||
            PySequence_Length(s) < 0) {
        PyErr_Clear();
        return 0;
    }
    if (PyArray_Check(s)) {
        return PyArray_NDIM(s);
    }
    if (PyArray_IsScalar(s, Generic)) {
        return 0;
    }
    if (PyString_Check(s) || PyBuffer_Check(s) || PyUnicode_Check(s)) {
        return stop_at_string ? 0:1;
    }
    if (stop_at_tuple && PyTuple_Check(s)) {
        return 0;
    }
    if ((e = PyObject_GetAttrString(s, "__array_struct__")) != NULL) {
        d = -1;
        if (PyCObject_Check(e)) {
            PyArrayInterface *inter;
            inter = (PyArrayInterface *)PyCObject_AsVoidPtr(e);
            if (inter->two == 2) {
                d = inter->nd;
            }
        }
        Py_DECREF(e);
        if (d > -1) {
            return d;
        }
    }
    else {
        PyErr_Clear();
    }
    if ((e=PyObject_GetAttrString(s, "__array_interface__")) != NULL) {
        d = -1;
        if (PyDict_Check(e)) {
            PyObject *new;
            new = PyDict_GetItemString(e, "shape");
            if (new && PyTuple_Check(new)) {
                d = PyTuple_GET_SIZE(new);
            }
        }
        Py_DECREF(e);
        if (d>-1) {
            return d;
        }
    }
    else PyErr_Clear();

    if (PySequence_Length(s) == 0) {
        return 1;
    }
    if ((e=PySequence_GetItem(s,0)) == NULL) {
        return -1;
    }
    if (e != s) {
        d = discover_depth(e, max-1, stop_at_string, stop_at_tuple);
        if (d >= 0) {
            d++;
        }
    }
    Py_DECREF(e);
    return d;
}

static int
discover_itemsize(PyObject *s, int nd, int *itemsize)
{
    int n, r, i;
    PyObject *e;

    if (PyArray_Check(s)) {
	*itemsize = MAX(*itemsize, PyArray_ITEMSIZE(s));
	return 0;
    }

    n = PyObject_Length(s);
    if ((nd == 0) || PyString_Check(s) ||
            PyUnicode_Check(s) || PyBuffer_Check(s)) {
        *itemsize = MAX(*itemsize, n);
        return 0;
    }
    for (i = 0; i < n; i++) {
        if ((e = PySequence_GetItem(s,i))==NULL) {
            return -1;
        }
        r = discover_itemsize(e,nd-1,itemsize);
        Py_DECREF(e);
        if (r == -1) {
            return -1;
        }
    }
    return 0;
}

/*
 * Take an arbitrary object known to represent
 * an array of ndim nd, and determine the size in each dimension
 */
static int
discover_dimensions(PyObject *s, int nd, intp *d, int check_it)
{
    PyObject *e;
    int r, n, i, n_lower;


    if (PyArray_Check(s)) {
	for (i=0; i<nd; i++) {
	    d[i] = PyArray_DIM(s,i);
	}
	return 0;
    }
    n = PyObject_Length(s);
    *d = n;
    if (*d < 0) {
        return -1;
    }
    if (nd <= 1) {
        return 0;
    }
    n_lower = 0;
    for(i = 0; i < n; i++) {
        if ((e = PySequence_GetItem(s,i)) == NULL) {
            return -1;
        }
        r = discover_dimensions(e, nd - 1, d + 1, check_it);
        Py_DECREF(e);

        if (r == -1) {
            return -1;
        }
        if (check_it && n_lower != 0 && n_lower != d[1]) {
            PyErr_SetString(PyExc_ValueError,
                            "inconsistent shape in sequence");
            return -1;
        }
        if (d[1] > n_lower) {
            n_lower = d[1];
        }
    }
    d[1] = n_lower;

    return 0;
}

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

static PyArray_Descr *
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
    type = (PyObject *)op->ob_type;
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
static PyArray_Descr *
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

    if (PyBuffer_Check(op)) {
        chktype = PyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = op->ob_type->tp_as_sequence->sq_length(op);
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

    if (PyInstance_Check(op)) {
        goto deflt;
    }
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

/* adapted from Numarray */
static int
setArrayFromSequence(PyArrayObject *a, PyObject *s, int dim, intp offset)
{
    Py_ssize_t i, slen;
    int res = 0;

    /*
     * This code is to ensure that the sequence access below will
     * return a lower-dimensional sequence.
     */
    if (PyArray_Check(s) && !(PyArray_CheckExact(s))) {
      /*
       * FIXME:  This could probably copy the entire subarray at once here using
       * a faster algorithm.  Right now, just make sure a base-class array is
       * used so that the dimensionality reduction assumption is correct.
       */
	s = PyArray_EnsureArray(s);
    }

    if (dim > a->nd) {
        PyErr_Format(PyExc_ValueError,
                     "setArrayFromSequence: sequence/array dimensions mismatch.");
        return -1;
    }

    slen = PySequence_Length(s);
    if (slen != a->dimensions[dim]) {
        PyErr_Format(PyExc_ValueError,
                     "setArrayFromSequence: sequence/array shape mismatch.");
        return -1;
    }

    for (i = 0; i < slen; i++) {
        PyObject *o = PySequence_GetItem(s, i);
        if ((a->nd - dim) > 1) {
            res = setArrayFromSequence(a, o, dim+1, offset);
        }
        else {
            res = a->descr->f->setitem(o, (a->data + offset), a);
        }
        Py_DECREF(o);
        if (res < 0) {
            return res;
        }
        offset += a->strides[dim];
    }
    return 0;
}


static int
Assign_Array(PyArrayObject *self, PyObject *v)
{
    if (!PySequence_Check(v)) {
        PyErr_SetString(PyExc_ValueError,
                        "assignment from non-sequence");
        return -1;
    }
    if (self->nd == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "assignment to 0-d array");
        return -1;
    }
    return setArrayFromSequence(self, v, 0, 0);
}

/*
 * "Array Scalars don't call this code"
 * steals reference to typecode -- no NULL
 */
static PyObject *
Array_FromPyScalar(PyObject *op, PyArray_Descr *typecode)
{
    PyArrayObject *ret;
    int itemsize;
    int type;

    itemsize = typecode->elsize;
    type = typecode->type_num;

    if (itemsize == 0 && PyTypeNum_ISEXTENDED(type)) {
        itemsize = PyObject_Length(op);
        if (type == PyArray_UNICODE) {
            itemsize *= 4;
        }
        if (itemsize != typecode->elsize) {
            PyArray_DESCR_REPLACE(typecode);
            typecode->elsize = itemsize;
        }
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, typecode,
                                                0, NULL,
                                                NULL, NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (ret->nd > 0) {
        PyErr_SetString(PyExc_ValueError,
                        "shape-mismatch on array construction");
        Py_DECREF(ret);
        return NULL;
    }

    ret->descr->f->setitem(op, ret->data, ret);
    if (PyErr_Occurred()) {
        Py_DECREF(ret);
        return NULL;
    }
    else {
        return (PyObject *)ret;
    }
}


/*
 * If s is not a list, return 0
 * Otherwise:
 *
 * run object_depth_and_dimension on all the elements
 * and make sure the returned shape and size is the
 * same for each element
 */
static int
object_depth_and_dimension(PyObject *s, int max, intp *dims)
{
    intp *newdims, *test_dims;
    int nd, test_nd;
    int i, islist, istuple;
    intp size;
    PyObject *obj;

    islist = PyList_Check(s);
    istuple = PyTuple_Check(s);
    if (!(islist || istuple)) {
        return 0;
    }

    size = PySequence_Size(s);
    if (size == 0) {
        return 0;
    }
    if (max < 1) {
        return 0;
    }
    if (max < 2) {
        dims[0] = size;
        return 1;
    }

    newdims = PyDimMem_NEW(2*(max - 1));
    test_dims = newdims + (max - 1);
    if (islist) {
        obj = PyList_GET_ITEM(s, 0);
    }
    else {
        obj = PyTuple_GET_ITEM(s, 0);
    }
    nd = object_depth_and_dimension(obj, max - 1, newdims);

    for (i = 1; i < size; i++) {
        if (islist) {
            obj = PyList_GET_ITEM(s, i);
        }
        else {
            obj = PyTuple_GET_ITEM(s, i);
        }
        test_nd = object_depth_and_dimension(obj, max-1, test_dims);

        if ((nd != test_nd) ||
            (!PyArray_CompareLists(newdims, test_dims, nd))) {
            nd = 0;
            break;
        }
    }

    for (i = 1; i <= nd; i++) {
        dims[i] = newdims[i-1];
    }
    dims[0] = size;
    PyDimMem_FREE(newdims);
    return nd + 1;
}

static PyObject *
ObjectArray_FromNestedList(PyObject *s, PyArray_Descr *typecode, int fortran)
{
    int nd;
    intp d[MAX_DIMS];
    PyArrayObject *r;

    /* Get the depth and the number of dimensions */
    nd = object_depth_and_dimension(s, MAX_DIMS, d);
    if (nd < 0) {
        return NULL;
    }
    if (nd == 0) {
        return Array_FromPyScalar(s, typecode);
    }
    r = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, typecode,
                                             nd, d,
                                             NULL, NULL,
                                             fortran, NULL);
    if (!r) {
        return NULL;
    }
    if(Assign_Array(r,s) == -1) {
        Py_DECREF(r);
        return NULL;
    }
    return (PyObject*)r;
}

/*
 * isobject means that we are constructing an
 * object array on-purpose with a nested list.
 * Only a list is interpreted as a sequence with these rules
 * steals reference to typecode
 */
static PyObject *
Array_FromSequence(PyObject *s, PyArray_Descr *typecode, int fortran,
                   int min_depth, int max_depth)
{
    PyArrayObject *r;
    int nd;
    int err;
    intp d[MAX_DIMS];
    int stop_at_string;
    int stop_at_tuple;
    int check_it;
    int type = typecode->type_num;
    int itemsize = typecode->elsize;

    check_it = (typecode->type != PyArray_CHARLTR);
    stop_at_string = (type != PyArray_STRING) ||
                     (typecode->type == PyArray_STRINGLTR);
    stop_at_tuple = (type == PyArray_VOID && (typecode->names
                                              || typecode->subarray));

    nd = discover_depth(s, MAX_DIMS + 1, stop_at_string, stop_at_tuple);
    if (nd == 0) {
        return Array_FromPyScalar(s, typecode);
    }
    else if (nd < 0) {
        PyErr_SetString(PyExc_ValueError,
                "invalid input sequence");
        goto fail;
    }
    if (max_depth && PyTypeNum_ISOBJECT(type) && (nd > max_depth)) {
        nd = max_depth;
    }
    if ((max_depth && nd > max_depth) || (min_depth && nd < min_depth)) {
        PyErr_SetString(PyExc_ValueError,
                "invalid number of dimensions");
        goto fail;
    }

    err = discover_dimensions(s, nd, d, check_it);
    if (err == -1) {
        goto fail;
    }
    if (typecode->type == PyArray_CHARLTR && nd > 0 && d[nd - 1] == 1) {
        nd = nd - 1;
    }

    if (itemsize == 0 && PyTypeNum_ISEXTENDED(type)) {
        err = discover_itemsize(s, nd, &itemsize);
        if (err == -1) {
            goto fail;
        }
        if (type == PyArray_UNICODE) {
            itemsize *= 4;
        }
    }
    if (itemsize != typecode->elsize) {
        PyArray_DESCR_REPLACE(typecode);
        typecode->elsize = itemsize;
    }

    r = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, typecode,
                                             nd, d,
                                             NULL, NULL,
                                             fortran, NULL);
    if (!r) {
        return NULL;
    }

    err = Assign_Array(r,s);
    if (err == -1) {
        Py_DECREF(r);
        return NULL;
    }
    return (PyObject*)r;

 fail:
    Py_DECREF(typecode);
    return NULL;
}


/*NUMPY_API
 * Is the typenum valid?
 */
NPY_NO_EXPORT int
PyArray_ValidType(int type)
{
    PyArray_Descr *descr;
    int res=TRUE;

    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        res = FALSE;
    }
    Py_DECREF(descr);
    return res;
}

/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to at --- cannot be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *mp, PyArray_Descr *at, int fortran)
{
    PyObject *out;
    int ret;
    PyArray_Descr *mpd;

    mpd = mp->descr;

    if (((mpd == at) ||
                ((mpd->type_num == at->type_num) &&
                 PyArray_EquivByteorders(mpd->byteorder, at->byteorder) &&
                 ((mpd->elsize == at->elsize) || (at->elsize==0)))) &&
                 PyArray_ISBEHAVED_RO(mp)) {
        Py_DECREF(at);
        Py_INCREF(mp);
        return (PyObject *)mp;
    }

    if (at->elsize == 0) {
        PyArray_DESCR_REPLACE(at);
        if (at == NULL) {
            return NULL;
        }
        if (mpd->type_num == PyArray_STRING &&
            at->type_num == PyArray_UNICODE) {
            at->elsize = mpd->elsize << 2;
        }
        if (mpd->type_num == PyArray_UNICODE &&
            at->type_num == PyArray_STRING) {
            at->elsize = mpd->elsize >> 2;
        }
        if (at->type_num == PyArray_VOID) {
            at->elsize = mpd->elsize;
        }
    }

    out = PyArray_NewFromDescr(mp->ob_type, at,
                               mp->nd,
                               mp->dimensions,
                               NULL, NULL,
                               fortran,
                               (PyObject *)mp);

    if (out == NULL) {
        return NULL;
    }
    ret = PyArray_CastTo((PyArrayObject *)out, mp);
    if (ret != -1) {
        return out;
    }

    Py_DECREF(out);
    return NULL;

}

/*NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num)
{
    PyArray_VectorUnaryFunc *castfunc = NULL;

    if (type_num < PyArray_NTYPES) {
        castfunc = descr->f->cast[type_num];
    }
    if (castfunc == NULL) {
        PyObject *obj = descr->f->castdict;
        if (obj && PyDict_Check(obj)) {
            PyObject *key;
            PyObject *cobj;

            key = PyInt_FromLong(type_num);
            cobj = PyDict_GetItem(obj, key);
            Py_DECREF(key);
            if (PyCObject_Check(cobj)) {
                castfunc = PyCObject_AsVoidPtr(cobj);
            }
        }
        if (castfunc) {
            return castfunc;
        }
    }
    else {
        return castfunc;
    }

    PyErr_SetString(PyExc_ValueError, "No cast function available.");
    return NULL;
}

/*
 * Reference counts:
 * copyswapn is used which increases and decreases reference counts for OBJECT arrays.
 * All that needs to happen is for any reference counts in the buffers to be
 * decreased when completely finished with the buffers.
 *
 * buffers[0] is the destination
 * buffers[1] is the source
 */
static void
_strided_buffered_cast(char *dptr, intp dstride, int delsize, int dswap,
                       PyArray_CopySwapNFunc *dcopyfunc,
                       char *sptr, intp sstride, int selsize, int sswap,
                       PyArray_CopySwapNFunc *scopyfunc,
                       intp N, char **buffers, int bufsize,
                       PyArray_VectorUnaryFunc *castfunc,
                       PyArrayObject *dest, PyArrayObject *src)
{
    int i;
    if (N <= bufsize) {
        /*
         * 1. copy input to buffer and swap
         * 2. cast input to output
         * 3. swap output if necessary and copy from output buffer
         */
        scopyfunc(buffers[1], selsize, sptr, sstride, N, sswap, src);
        castfunc(buffers[1], buffers[0], N, src, dest);
        dcopyfunc(dptr, dstride, buffers[0], delsize, N, dswap, dest);
        return;
    }

    /* otherwise we need to divide up into bufsize pieces */
    i = 0;
    while (N > 0) {
        int newN = MIN(N, bufsize);

        _strided_buffered_cast(dptr+i*dstride, dstride, delsize,
                               dswap, dcopyfunc,
                               sptr+i*sstride, sstride, selsize,
                               sswap, scopyfunc,
                               newN, buffers, bufsize, castfunc, dest, src);
        i += newN;
        N -= bufsize;
    }
    return;
}

static int
_broadcast_cast(PyArrayObject *out, PyArrayObject *in,
                PyArray_VectorUnaryFunc *castfunc, int iswap, int oswap)
{
    int delsize, selsize, maxaxis, i, N;
    PyArrayMultiIterObject *multi;
    intp maxdim, ostrides, istrides;
    char *buffers[2];
    PyArray_CopySwapNFunc *ocopyfunc, *icopyfunc;
    char *obptr;
    NPY_BEGIN_THREADS_DEF;

    delsize = PyArray_ITEMSIZE(out);
    selsize = PyArray_ITEMSIZE(in);
    multi = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, out, in);
    if (multi == NULL) {
        return -1;
    }

    if (multi->size != PyArray_SIZE(out)) {
        PyErr_SetString(PyExc_ValueError,
                "array dimensions are not "\
                "compatible for copy");
        Py_DECREF(multi);
        return -1;
    }

    icopyfunc = in->descr->f->copyswapn;
    ocopyfunc = out->descr->f->copyswapn;
    maxaxis = PyArray_RemoveSmallest(multi);
    if (maxaxis < 0) {
        /* cast 1 0-d array to another */
        N = 1;
        maxdim = 1;
        ostrides = delsize;
        istrides = selsize;
    }
    else {
        maxdim = multi->dimensions[maxaxis];
        N = (int) (MIN(maxdim, PyArray_BUFSIZE));
        ostrides = multi->iters[0]->strides[maxaxis];
        istrides = multi->iters[1]->strides[maxaxis];

    }
    buffers[0] = _pya_malloc(N*delsize);
    if (buffers[0] == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    buffers[1] = _pya_malloc(N*selsize);
    if (buffers[1] == NULL) {
        _pya_free(buffers[0]);
        PyErr_NoMemory();
        return -1;
    }
    if (PyDataType_FLAGCHK(out->descr, NPY_NEEDS_INIT)) {
        memset(buffers[0], 0, N*delsize);
    }
    if (PyDataType_FLAGCHK(in->descr, NPY_NEEDS_INIT)) {
        memset(buffers[1], 0, N*selsize);
    }

#if NPY_ALLOW_THREADS
    if (PyArray_ISNUMBER(in) && PyArray_ISNUMBER(out)) {
        NPY_BEGIN_THREADS;
    }
#endif

    while (multi->index < multi->size) {
        _strided_buffered_cast(multi->iters[0]->dataptr,
                ostrides,
                delsize, oswap, ocopyfunc,
                multi->iters[1]->dataptr,
                istrides,
                selsize, iswap, icopyfunc,
                maxdim, buffers, N,
                castfunc, out, in);
        PyArray_MultiIter_NEXT(multi);
    }
#if NPY_ALLOW_THREADS
    if (PyArray_ISNUMBER(in) && PyArray_ISNUMBER(out)) {
        NPY_END_THREADS;
    }
#endif
    Py_DECREF(multi);
    if (PyDataType_REFCHK(in->descr)) {
        obptr = buffers[1];
        for (i = 0; i < N; i++, obptr+=selsize) {
            PyArray_Item_XDECREF(obptr, out->descr);
        }
    }
    if (PyDataType_REFCHK(out->descr)) {
        obptr = buffers[0];
        for (i = 0; i < N; i++, obptr+=delsize) {
            PyArray_Item_XDECREF(obptr, out->descr);
        }
    }
    _pya_free(buffers[0]);
    _pya_free(buffers[1]);
    if (PyErr_Occurred()) {
        return -1;
    }

    return 0;
}



/*
 * Must be broadcastable.
 * This code is very similar to PyArray_CopyInto/PyArray_MoveInto
 * except casting is done --- PyArray_BUFSIZE is used
 * as the size of the casting buffer.
 */

/*NUMPY_API
 * Cast to an already created array.
 */
NPY_NO_EXPORT int
PyArray_CastTo(PyArrayObject *out, PyArrayObject *mp)
{
    int simple;
    int same;
    PyArray_VectorUnaryFunc *castfunc = NULL;
    int mpsize = PyArray_SIZE(mp);
    int iswap, oswap;
    NPY_BEGIN_THREADS_DEF;

    if (mpsize == 0) {
        return 0;
    }
    if (!PyArray_ISWRITEABLE(out)) {
        PyErr_SetString(PyExc_ValueError, "output array is not writeable");
        return -1;
    }

    castfunc = PyArray_GetCastFunc(mp->descr, out->descr->type_num);
    if (castfunc == NULL) {
        return -1;
    }

    same = PyArray_SAMESHAPE(out, mp);
    simple = same && ((PyArray_ISCARRAY_RO(mp) && PyArray_ISCARRAY(out)) ||
            (PyArray_ISFARRAY_RO(mp) && PyArray_ISFARRAY(out)));
    if (simple) {
#if NPY_ALLOW_THREADS
        if (PyArray_ISNUMBER(mp) && PyArray_ISNUMBER(out)) {
            NPY_BEGIN_THREADS;
        }
#endif
        castfunc(mp->data, out->data, mpsize, mp, out);

#if NPY_ALLOW_THREADS
        if (PyArray_ISNUMBER(mp) && PyArray_ISNUMBER(out)) {
            NPY_END_THREADS;
        }
#endif
        if (PyErr_Occurred()) {
            return -1;
        }
        return 0;
    }

    /*
     * If the input or output is OBJECT, STRING, UNICODE, or VOID
     *  then getitem and setitem are used for the cast
     *  and byteswapping is handled by those methods
     */
    if (PyArray_ISFLEXIBLE(mp) || PyArray_ISOBJECT(mp) || PyArray_ISOBJECT(out) ||
            PyArray_ISFLEXIBLE(out)) {
        iswap = oswap = 0;
    }
    else {
        iswap = PyArray_ISBYTESWAPPED(mp);
        oswap = PyArray_ISBYTESWAPPED(out);
    }

    return _broadcast_cast(out, mp, castfunc, iswap, oswap);
}


static int
_bufferedcast(PyArrayObject *out, PyArrayObject *in,
              PyArray_VectorUnaryFunc *castfunc)
{
    char *inbuffer, *bptr, *optr;
    char *outbuffer=NULL;
    PyArrayIterObject *it_in = NULL, *it_out = NULL;
    intp i, index;
    intp ncopies = PyArray_SIZE(out) / PyArray_SIZE(in);
    int elsize=in->descr->elsize;
    int nels = PyArray_BUFSIZE;
    int el;
    int inswap, outswap = 0;
    int obuf=!PyArray_ISCARRAY(out);
    int oelsize = out->descr->elsize;
    PyArray_CopySwapFunc *in_csn;
    PyArray_CopySwapFunc *out_csn;
    int retval = -1;

    in_csn = in->descr->f->copyswap;
    out_csn = out->descr->f->copyswap;

    /*
     * If the input or output is STRING, UNICODE, or VOID
     * then getitem and setitem are used for the cast
     *  and byteswapping is handled by those methods
     */

    inswap = !(PyArray_ISFLEXIBLE(in) || PyArray_ISNOTSWAPPED(in));

    inbuffer = PyDataMem_NEW(PyArray_BUFSIZE*elsize);
    if (inbuffer == NULL) {
        return -1;
    }
    if (PyArray_ISOBJECT(in)) {
        memset(inbuffer, 0, PyArray_BUFSIZE*elsize);
    }
    it_in = (PyArrayIterObject *)PyArray_IterNew((PyObject *)in);
    if (it_in == NULL) {
        goto exit;
    }
    if (obuf) {
        outswap = !(PyArray_ISFLEXIBLE(out) ||
                    PyArray_ISNOTSWAPPED(out));
        outbuffer = PyDataMem_NEW(PyArray_BUFSIZE*oelsize);
        if (outbuffer == NULL) {
            goto exit;
        }
        if (PyArray_ISOBJECT(out)) {
            memset(outbuffer, 0, PyArray_BUFSIZE*oelsize);
        }
        it_out = (PyArrayIterObject *)PyArray_IterNew((PyObject *)out);
        if (it_out == NULL) {
            goto exit;
        }
        nels = MIN(nels, PyArray_BUFSIZE);
    }

    optr = (obuf) ? outbuffer: out->data;
    bptr = inbuffer;
    el = 0;
    while (ncopies--) {
        index = it_in->size;
        PyArray_ITER_RESET(it_in);
        while (index--) {
            in_csn(bptr, it_in->dataptr, inswap, in);
            bptr += elsize;
            PyArray_ITER_NEXT(it_in);
            el += 1;
            if ((el == nels) || (index == 0)) {
                /* buffer filled, do cast */
                castfunc(inbuffer, optr, el, in, out);
                if (obuf) {
                    /* Copy from outbuffer to array */
                    for (i = 0; i < el; i++) {
                        out_csn(it_out->dataptr,
                                optr, outswap,
                                out);
                        optr += oelsize;
                        PyArray_ITER_NEXT(it_out);
                    }
                    optr = outbuffer;
                }
                else {
                    optr += out->descr->elsize * nels;
                }
                el = 0;
                bptr = inbuffer;
            }
        }
    }
    retval = 0;

 exit:
    Py_XDECREF(it_in);
    PyDataMem_FREE(inbuffer);
    PyDataMem_FREE(outbuffer);
    if (obuf) {
        Py_XDECREF(it_out);
    }
    return retval;
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
NPY_NO_EXPORT int
PyArray_CastAnyTo(PyArrayObject *out, PyArrayObject *mp)
{
    int simple;
    PyArray_VectorUnaryFunc *castfunc = NULL;
    int mpsize = PyArray_SIZE(mp);

    if (mpsize == 0) {
        return 0;
    }
    if (!PyArray_ISWRITEABLE(out)) {
        PyErr_SetString(PyExc_ValueError, "output array is not writeable");
        return -1;
    }

    if (!(mpsize == PyArray_SIZE(out))) {
        PyErr_SetString(PyExc_ValueError,
                        "arrays must have the same number of"
                        " elements for the cast.");
        return -1;
    }

    castfunc = PyArray_GetCastFunc(mp->descr, out->descr->type_num);
    if (castfunc == NULL) {
        return -1;
    }
    simple = ((PyArray_ISCARRAY_RO(mp) && PyArray_ISCARRAY(out)) ||
              (PyArray_ISFARRAY_RO(mp) && PyArray_ISFARRAY(out)));
    if (simple) {
        castfunc(mp->data, out->data, mpsize, mp, out);
        return 0;
    }
    if (PyArray_SAMESHAPE(out, mp)) {
        int iswap, oswap;
        iswap = PyArray_ISBYTESWAPPED(mp) && !PyArray_ISFLEXIBLE(mp);
        oswap = PyArray_ISBYTESWAPPED(out) && !PyArray_ISFLEXIBLE(out);
        return _broadcast_cast(out, mp, castfunc, iswap, oswap);
    }
    return _bufferedcast(out, mp, castfunc);
}



/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags)
{

    PyArrayObject *ret = NULL;
    int itemsize;
    int copy = 0;
    int arrflags;
    PyArray_Descr *oldtype;
    char *msg = "cannot copy back to a read-only array";
    PyTypeObject *subtype;

    oldtype = PyArray_DESCR(arr);
    subtype = arr->ob_type;
    if (newtype == NULL) {
        newtype = oldtype; Py_INCREF(oldtype);
    }
    itemsize = newtype->elsize;
    if (itemsize == 0) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
        itemsize = newtype->elsize;
    }

    /*
     * Can't cast unless ndim-0 array, FORCECAST is specified
     * or the cast is safe.
     */
    if (!(flags & FORCECAST) && !PyArray_NDIM(arr) == 0 &&
        !PyArray_CanCastTo(oldtype, newtype)) {
        Py_DECREF(newtype);
        PyErr_SetString(PyExc_TypeError,
                        "array cannot be safely cast "  \
                        "to required type");
        return NULL;
    }

    /* Don't copy if sizes are compatible */
    if ((flags & ENSURECOPY) || PyArray_EquivTypes(oldtype, newtype)) {
        arrflags = arr->flags;
        copy = (flags & ENSURECOPY) ||
            ((flags & CONTIGUOUS) && (!(arrflags & CONTIGUOUS)))
            || ((flags & ALIGNED) && (!(arrflags & ALIGNED)))
            || (arr->nd > 1 &&
                ((flags & FORTRAN) && (!(arrflags & FORTRAN))))
            || ((flags & WRITEABLE) && (!(arrflags & WRITEABLE)));

        if (copy) {
            if ((flags & UPDATEIFCOPY) &&
                (!PyArray_ISWRITEABLE(arr))) {
                Py_DECREF(newtype);
                PyErr_SetString(PyExc_ValueError, msg);
                return NULL;
            }
            if ((flags & ENSUREARRAY)) {
                subtype = &PyArray_Type;
            }
            ret = (PyArrayObject *)
                PyArray_NewFromDescr(subtype, newtype,
                                     arr->nd,
                                     arr->dimensions,
                                     NULL, NULL,
                                     flags & FORTRAN,
                                     (PyObject *)arr);
            if (ret == NULL) {
                return NULL;
            }
            if (PyArray_CopyInto(ret, arr) == -1) {
                Py_DECREF(ret);
                return NULL;
            }
            if (flags & UPDATEIFCOPY)  {
                ret->flags |= UPDATEIFCOPY;
                ret->base = (PyObject *)arr;
                PyArray_FLAGS(ret->base) &= ~WRITEABLE;
                Py_INCREF(arr);
            }
        }
        /*
         * If no copy then just increase the reference
         * count and return the input
         */
        else {
            Py_DECREF(newtype);
            if ((flags & ENSUREARRAY) &&
                !PyArray_CheckExact(arr)) {
                Py_INCREF(arr->descr);
                ret = (PyArrayObject *)
                    PyArray_NewFromDescr(&PyArray_Type,
                                         arr->descr,
                                         arr->nd,
                                         arr->dimensions,
                                         arr->strides,
                                         arr->data,
                                         arr->flags,NULL);
                if (ret == NULL) {
                    return NULL;
                }
                ret->base = (PyObject *)arr;
            }
            else {
                ret = arr;
            }
            Py_INCREF(arr);
        }
    }

    /*
     * The desired output type is different than the input
     * array type and copy was not specified
     */
    else {
        if ((flags & UPDATEIFCOPY) &&
            (!PyArray_ISWRITEABLE(arr))) {
            Py_DECREF(newtype);
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        if ((flags & ENSUREARRAY)) {
            subtype = &PyArray_Type;
        }
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(subtype, newtype,
                                 arr->nd, arr->dimensions,
                                 NULL, NULL,
                                 flags & FORTRAN,
                                 (PyObject *)arr);
        if (ret == NULL) {
            return NULL;
        }
        if (PyArray_CastTo(ret, arr) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        if (flags & UPDATEIFCOPY)  {
            ret->flags |= UPDATEIFCOPY;
            ret->base = (PyObject *)arr;
            PyArray_FLAGS(ret->base) &= ~WRITEABLE;
            Py_INCREF(arr);
        }
    }
    return (PyObject *)ret;
}

/* new reference */
static PyArray_Descr *
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

/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    PyArray_Descr *thetype = NULL;
    char buf[40];
    PyArrayInterface *inter;
    PyObject *attr, *r;
    char endian = PyArray_NATBYTE;

    attr = PyObject_GetAttrString(input, "__array_struct__");
    if (attr == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!PyCObject_Check(attr)) {
        goto fail;
    }
    inter = PyCObject_AsVoidPtr(attr);
    if (inter->two != 2) {
        goto fail;
    }
    if ((inter->flags & NOTSWAPPED) != NOTSWAPPED) {
        endian = PyArray_OPPBYTE;
        inter->flags &= ~NOTSWAPPED;
    }

    if (inter->flags & ARR_HAS_DESCR) {
        if (PyArray_DescrConverter(inter->descr, &thetype) == PY_FAIL) {
            thetype = NULL;
            PyErr_Clear();
        }
    }

    if (thetype == NULL) {
        PyOS_snprintf(buf, sizeof(buf),
                "%c%c%d", endian, inter->typekind, inter->itemsize);
        if (!(thetype=_array_typedescr_fromstr(buf))) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    r = PyArray_NewFromDescr(&PyArray_Type, thetype,
                             inter->nd, inter->shape,
                             inter->strides, inter->data,
                             inter->flags, NULL);
    Py_INCREF(input);
    PyArray_BASE(r) = input;
    Py_DECREF(attr);
    PyArray_UpdateFlags((PyArrayObject *)r, UPDATE_ALL);
    return r;

 fail:
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    Py_DECREF(attr);
    return NULL;
}

#define PyIntOrLong_Check(obj) (PyInt_Check(obj) || PyLong_Check(obj))

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input)
{
    PyObject *attr = NULL, *item = NULL;
    PyObject *tstr = NULL, *shape = NULL;
    PyObject *inter = NULL;
    PyObject *base = NULL;
    PyArrayObject *ret;
    PyArray_Descr *type=NULL;
    char *data;
    Py_ssize_t buffer_len;
    int res, i, n;
    intp dims[MAX_DIMS], strides[MAX_DIMS];
    int dataflags = BEHAVED;

    /* Get the memory from __array_data__ and __array_offset__ */
    /* Get the shape */
    /* Get the typestring -- ignore array_descr */
    /* Get the strides */

    inter = PyObject_GetAttrString(input, "__array_interface__");
    if (inter == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!PyDict_Check(inter)) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    shape = PyDict_GetItemString(inter, "shape");
    if (shape == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    tstr = PyDict_GetItemString(inter, "typestr");
    if (tstr == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }

    attr = PyDict_GetItemString(inter, "data");
    base = input;
    if ((attr == NULL) || (attr==Py_None) || (!PyTuple_Check(attr))) {
        if (attr && (attr != Py_None)) {
            item = attr;
        }
        else {
            item = input;
        }
        res = PyObject_AsWriteBuffer(item, (void **)&data, &buffer_len);
        if (res < 0) {
            PyErr_Clear();
            res = PyObject_AsReadBuffer(
                    item, (const void **)&data, &buffer_len);
            if (res < 0) {
                goto fail;
            }
            dataflags &= ~WRITEABLE;
        }
        attr = PyDict_GetItemString(inter, "offset");
        if (attr) {
            longlong num = PyLong_AsLongLong(attr);
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                                "offset "\
                                "must be an integer");
                goto fail;
            }
            data += num;
        }
        base = item;
    }
    else {
        PyObject *dataptr;
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                            "data must return "     \
                            "a 2-tuple with (data pointer "\
                            "integer, read-only flag)");
            goto fail;
        }
        dataptr = PyTuple_GET_ITEM(attr, 0);
        if (PyString_Check(dataptr)) {
            res = sscanf(PyString_AsString(dataptr),
                         "%p", (void **)&data);
            if (res < 1) {
                PyErr_SetString(PyExc_TypeError,
                                "data string cannot be " \
                                "converted");
                goto fail;
            }
        }
        else if (PyIntOrLong_Check(dataptr)) {
            data = PyLong_AsVoidPtr(dataptr);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "first element " \
                            "of data tuple must be integer" \
                            " or string.");
            goto fail;
        }
        if (PyObject_IsTrue(PyTuple_GET_ITEM(attr,1))) {
            dataflags &= ~WRITEABLE;
        }
    }
    attr = tstr;
    if (!PyString_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "typestr must be a string");
        goto fail;
    }
    type = _array_typedescr_fromstr(PyString_AS_STRING(attr));
    if (type == NULL) {
        goto fail;
    }
    attr = shape;
    if (!PyTuple_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
        Py_DECREF(type);
        goto fail;
    }
    n = PyTuple_GET_SIZE(attr);
    for (i = 0; i < n; i++) {
        item = PyTuple_GET_ITEM(attr, i);
        dims[i] = PyArray_PyIntAsIntp(item);
        if (error_converting(dims[i])) {
            break;
        }
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, type,
                                                n, dims,
                                                NULL, data,
                                                dataflags, NULL);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(base);
    ret->base = base;

    attr = PyDict_GetItemString(inter, "strides");
    if (attr != NULL && attr != Py_None) {
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                            "strides must be a tuple");
            Py_DECREF(ret);
            return NULL;
        }
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                            "mismatch in length of "\
                            "strides and shape");
            Py_DECREF(ret);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            item = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(item);
            if (error_converting(strides[i])) {
                break;
            }
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        memcpy(ret->strides, strides, n*sizeof(intp));
    }
    else PyErr_Clear();
    PyArray_UpdateFlags(ret, UPDATE_ALL);
    Py_DECREF(inter);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(inter);
    return NULL;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    PyObject *new;
    PyObject *array_meth;

    array_meth = PyObject_GetAttrString(op, "__array__");
    if (array_meth == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (context == NULL) {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, NULL);
        }
        else {
            new = PyObject_CallFunction(array_meth, "O", typecode);
        }
    }
    else {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, "OO", Py_None, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "");
            }
        }
        else {
            new = PyObject_CallFunction(array_meth, "OO", typecode, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "O", typecode);
            }
        }
    }
    Py_DECREF(array_meth);
    if (new == NULL) {
        return NULL;
    }
    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not "  \
                        "producing an array");
        Py_DECREF(new);
        return NULL;
    }
    return new;
}

/*NUMPY_API
 * Does not check for ENSURECOPY and NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from lot's of different places which
     * is why there are so many checks.  The comments try to
     * explain some of the checks.
     */
    PyObject *r = NULL;
    int seq = FALSE;

    /*
     * Is input object already an array?
     * This is where the flags are used
     */
    if (PyArray_Check(op)) {
        r = PyArray_FromArray((PyArrayObject *)op, newtype, flags);
    }
    else if (PyArray_IsScalar(op, Generic)) {
        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        r = PyArray_FromScalar(op, newtype);
    }
    else if (newtype == NULL &&
               (newtype = _array_find_python_scalar_type(op))) {
        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        r = Array_FromPyScalar(op, newtype);
    }
    else if (PyArray_HasArrayInterfaceType(op, newtype, context, r)) {
        PyObject *new;
        if (r == NULL) {
            Py_XDECREF(newtype);
            return NULL;
        }
        if (newtype != NULL || flags != 0) {
            new = PyArray_FromArray((PyArrayObject *)r, newtype, flags);
            Py_DECREF(r);
            r = new;
        }
    }
    else {
        int isobject = 0;

        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        if (newtype == NULL) {
            newtype = _array_find_type(op, NULL, MAX_DIMS);
        }
        else if (newtype->type_num == PyArray_OBJECT) {
            isobject = 1;
        }
        if (PySequence_Check(op)) {
            PyObject *thiserr = NULL;

            /* necessary but not sufficient */
            Py_INCREF(newtype);
            r = Array_FromSequence(op, newtype, flags & FORTRAN,
                                   min_depth, max_depth);
            if (r == NULL && (thiserr=PyErr_Occurred())) {
                if (PyErr_GivenExceptionMatches(thiserr,
                                                PyExc_MemoryError)) {
                    return NULL;
                }
                /*
                 * If object was explicitly requested,
                 * then try nested list object array creation
                 */
                PyErr_Clear();
                if (isobject) {
                    Py_INCREF(newtype);
                    r = ObjectArray_FromNestedList
                        (op, newtype, flags & FORTRAN);
                    seq = TRUE;
                    Py_DECREF(newtype);
                }
            }
            else {
                seq = TRUE;
                Py_DECREF(newtype);
            }
        }
        if (!seq) {
            r = Array_FromPyScalar(op, newtype);
        }
    }

    /* If we didn't succeed return NULL */
    if (r == NULL) {
        return NULL;
    }

    /* Be sure we succeed here */
    if(!PyArray_Check(r)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "internal error: PyArray_FromAny "\
                        "not producing an array");
        Py_DECREF(r);
        return NULL;
    }

    if (min_depth != 0 && ((PyArrayObject *)r)->nd < min_depth) {
        PyErr_SetString(PyExc_ValueError,
                        "object of too small depth for desired array");
        Py_DECREF(r);
        return NULL;
    }
    if (max_depth != 0 && ((PyArrayObject *)r)->nd > max_depth) {
        PyErr_SetString(PyExc_ValueError,
                        "object too deep for desired array");
        Py_DECREF(r);
        return NULL;
    }
    return r;

 err:
    Py_XDECREF(newtype);
    PyErr_SetString(PyExc_TypeError,
                    "UPDATEIFCOPY used for non-array input.");
    return NULL;
}

/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
{
    return _array_find_type(op, mintype, MAX_DIMS);
}

/*NUMPY_API
 * Return the typecode of the array a Python object would be converted to
 */
NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *intype;
    PyArray_Descr *outtype;
    int ret;

    intype = PyArray_DescrFromType(minimum_type);
    if (intype == NULL) {
        PyErr_Clear();
    }
    outtype = _array_find_type(op, intype, MAX_DIMS);
    ret = outtype->type_num;
    Py_DECREF(outtype);
    Py_XDECREF(intype);
    return ret;
}


/*
 * flags is any of
 * CONTIGUOUS,
 * FORTRAN,
 * ALIGNED,
 * WRITEABLE,
 * NOTSWAPPED,
 * ENSURECOPY,
 * UPDATEIFCOPY,
 * FORCECAST,
 * ENSUREARRAY,
 * ELEMENTSTRIDES
 *
 * or'd (|) together
 *
 * Any of these flags present means that the returned array should
 * guarantee that aspect of the array.  Otherwise the returned array
 * won't guarantee it -- it will depend on the object as to whether or
 * not it has such features.
 *
 * Note that ENSURECOPY is enough
 * to guarantee CONTIGUOUS, ALIGNED and WRITEABLE
 * and therefore it is redundant to include those as well.
 *
 * BEHAVED == ALIGNED | WRITEABLE
 * CARRAY = CONTIGUOUS | BEHAVED
 * FARRAY = FORTRAN | BEHAVED
 *
 * FORTRAN can be set in the FLAGS to request a FORTRAN array.
 * Fortran arrays are always behaved (aligned,
 * notswapped, and writeable) and not (C) CONTIGUOUS (if > 1d).
 *
 * UPDATEIFCOPY flag sets this flag in the returned array if a copy is
 * made and the base argument points to the (possibly) misbehaved array.
 * When the new array is deallocated, the original array held in base
 * is updated with the contents of the new array.
 *
 * FORCECAST will cause a cast to occur regardless of whether or not
 * it is safe.
 */

/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    PyObject *obj;
    if (requires & NOTSWAPPED) {
        if (!descr && PyArray_Check(op) &&
            !PyArray_ISNBO(PyArray_DESCR(op)->byteorder)) {
            descr = PyArray_DescrNew(PyArray_DESCR(op));
        }
        else if (descr && !PyArray_ISNBO(descr->byteorder)) {
            PyArray_DESCR_REPLACE(descr);
        }
        if (descr) {
            descr->byteorder = PyArray_NATIVE;
        }
    }

    obj = PyArray_FromAny(op, descr, min_depth, max_depth, requires, context);
    if (obj == NULL) {
        return NULL;
    }
    if ((requires & ELEMENTSTRIDES) &&
        !PyArray_ElementStrides(obj)) {
        PyObject *new;
        new = PyArray_NewCopy((PyArrayObject *)obj, PyArray_ANYORDER);
        Py_DECREF(obj);
        obj = new;
    }
    return obj;
}

/*NUMPY_API
 * This is a quick wrapper around PyArray_FromAny(op, NULL, 0, 0, ENSUREARRAY)
 * that special cases Arrays and PyArray_Scalars up front
 * It *steals a reference* to the object
 * It also guarantees that the result is PyArray_Type
 * Because it decrefs op if any conversion needs to take place
 * so it can be used like PyArray_EnsureArray(some_function(...))
 */
NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op)
{
    PyObject *new;

    if (op == NULL) {
        return NULL;
    }
    if (PyArray_CheckExact(op)) {
        return op;
    }
    if (PyArray_Check(op)) {
        new = PyArray_View((PyArrayObject *)op, NULL, &PyArray_Type);
        Py_DECREF(op);
        return new;
    }
    if (PyArray_IsScalar(op, Generic)) {
        new = PyArray_FromScalar(op, NULL);
        Py_DECREF(op);
        return new;
    }
    new = PyArray_FromAny(op, NULL, 0, 0, ENSUREARRAY, NULL);
    Py_DECREF(op);
    return new;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op)
{
    if (op && PyArray_Check(op)) {
        return op;
    }
    return PyArray_EnsureArray(op);
}

/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    PyArray_Descr *from, *to;
    int felsize, telsize;

    if (fromtype == totype) {
        return 1;
    }
    if (fromtype == PyArray_BOOL) {
        return 1;
    }
    if (totype == PyArray_BOOL) {
        return 0;
    }
    if (totype == PyArray_OBJECT || totype == PyArray_VOID) {
        return 1;
    }
    if (fromtype == PyArray_OBJECT || fromtype == PyArray_VOID) {
        return 0;
    }
    from = PyArray_DescrFromType(fromtype);
    /*
     * cancastto is a PyArray_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    if (from->f->cancastto) {
        int *curtype;
        curtype = from->f->cancastto;
        while (*curtype != PyArray_NOTYPE) {
            if (*curtype++ == totype) {
                return 1;
            }
        }
    }
    if (PyTypeNum_ISUSERDEF(totype)) {
        return 0;
    }
    to = PyArray_DescrFromType(totype);
    telsize = to->elsize;
    felsize = from->elsize;
    Py_DECREF(from);
    Py_DECREF(to);

    switch(fromtype) {
    case PyArray_BYTE:
    case PyArray_SHORT:
    case PyArray_INT:
    case PyArray_LONG:
    case PyArray_LONGLONG:
        if (PyTypeNum_ISINTEGER(totype)) {
            if (PyTypeNum_ISUNSIGNED(totype)) {
                return 0;
            }
            else {
                return telsize >= felsize;
            }
        }
        else if (PyTypeNum_ISFLOAT(totype)) {
            if (felsize < 8) {
                return telsize > felsize;
            }
            else {
                return telsize >= felsize;
            }
        }
        else if (PyTypeNum_ISCOMPLEX(totype)) {
            if (felsize < 8) {
                return (telsize >> 1) > felsize;
            }
            else {
                return (telsize >> 1) >= felsize;
            }
        }
        else {
            return totype > fromtype;
        }
    case PyArray_UBYTE:
    case PyArray_USHORT:
    case PyArray_UINT:
    case PyArray_ULONG:
    case PyArray_ULONGLONG:
        if (PyTypeNum_ISINTEGER(totype)) {
            if (PyTypeNum_ISSIGNED(totype)) {
                return telsize > felsize;
            }
            else {
                return telsize >= felsize;
            }
        }
        else if (PyTypeNum_ISFLOAT(totype)) {
            if (felsize < 8) {
                return telsize > felsize;
            }
            else {
                return telsize >= felsize;
            }
        }
        else if (PyTypeNum_ISCOMPLEX(totype)) {
            if (felsize < 8) {
                return (telsize >> 1) > felsize;
            }
            else {
                return (telsize >> 1) >= felsize;
            }
        }
        else {
            return totype > fromtype;
        }
    case PyArray_FLOAT:
    case PyArray_DOUBLE:
    case PyArray_LONGDOUBLE:
        if (PyTypeNum_ISCOMPLEX(totype)) {
            return (telsize >> 1) >= felsize;
        }
        else {
            return totype > fromtype;
        }
    case PyArray_CFLOAT:
    case PyArray_CDOUBLE:
    case PyArray_CLONGDOUBLE:
        return totype > fromtype;
    case PyArray_STRING:
    case PyArray_UNICODE:
        return totype > fromtype;
    default:
        return 0;
    }
}

/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 */
NPY_NO_EXPORT Bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    int fromtype=from->type_num;
    int totype=to->type_num;
    Bool ret;

    ret = (Bool) PyArray_CanCastSafely(fromtype, totype);
    if (ret) {
        /* Check String and Unicode more closely */
        if (fromtype == PyArray_STRING) {
            if (totype == PyArray_STRING) {
                ret = (from->elsize <= to->elsize);
            }
            else if (totype == PyArray_UNICODE) {
                ret = (from->elsize << 2 <= to->elsize);
            }
        }
        else if (fromtype == PyArray_UNICODE) {
            if (totype == PyArray_UNICODE) {
                ret = (from->elsize <= to->elsize);
            }
        }
        /*
         * TODO: If totype is STRING or unicode
         * see if the length is long enough to hold the
         * stringified value of the object.
         */
    }
    return ret;
}

/*NUMPY_API
 * See if array scalars can be cast.
 */
NPY_NO_EXPORT Bool
PyArray_CanCastScalar(PyTypeObject *from, PyTypeObject *to)
{
    int fromtype;
    int totype;

    fromtype = _typenum_fromtypeobj((PyObject *)from, 0);
    totype = _typenum_fromtypeobj((PyObject *)to, 0);
    if (fromtype == PyArray_NOTYPE || totype == PyArray_NOTYPE) {
        return FALSE;
    }
    return (Bool) PyArray_CanCastSafely(fromtype, totype);
}

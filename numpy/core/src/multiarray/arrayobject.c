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

#include "common.h"

#include "arraytypes.h"
#include "scalartypes.h"
#include "arrayobject.h"
#include "ctors.h"
#include "methods.h"
#include "descriptor.h"
#include "iterators.h"
#include "mapping.h"
#include "getset.h"
#include "sequence.h"

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

#ifndef Py_UNICODE_WIDE
#include "ucsnarrow.c"
#endif


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

NPY_NO_EXPORT int
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

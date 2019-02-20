#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "ctors.h"
#include "descriptor.h"
#include "scalartypes.h"

#include "common.h"

static PyArray_Descr *
_descr_from_subtype(PyObject *type)
{
    PyObject *mro;
    mro = ((PyTypeObject *)type)->tp_mro;
    if (PyTuple_GET_SIZE(mro) < 2) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    return PyArray_DescrFromTypeObject(PyTuple_GET_ITEM(mro, 1));
}

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr)
{
    int type_num;
    int align;
    npy_intp memloc;
    if (descr == NULL) {
        descr = PyArray_DescrFromScalar(scalar);
        type_num = descr->type_num;
        Py_DECREF(descr);
    }
    else {
        type_num = descr->type_num;
    }
    switch (type_num) {
#define CASE(ut,lt) case NPY_##ut: return &(((Py##lt##ScalarObject *)scalar)->obval)
        CASE(BOOL, Bool);
        CASE(BYTE, Byte);
        CASE(UBYTE, UByte);
        CASE(SHORT, Short);
        CASE(USHORT, UShort);
        CASE(INT, Int);
        CASE(UINT, UInt);
        CASE(LONG, Long);
        CASE(ULONG, ULong);
        CASE(LONGLONG, LongLong);
        CASE(ULONGLONG, ULongLong);
        CASE(HALF, Half);
        CASE(FLOAT, Float);
        CASE(DOUBLE, Double);
        CASE(LONGDOUBLE, LongDouble);
        CASE(CFLOAT, CFloat);
        CASE(CDOUBLE, CDouble);
        CASE(CLONGDOUBLE, CLongDouble);
        CASE(OBJECT, Object);
        CASE(DATETIME, Datetime);
        CASE(TIMEDELTA, Timedelta);
#undef CASE
        case NPY_STRING:
            return (void *)PyString_AS_STRING(scalar);
        case NPY_UNICODE:
            return (void *)PyUnicode_AS_DATA(scalar);
        case NPY_VOID:
            return ((PyVoidScalarObject *)scalar)->obval;
    }

    /*
     * Must be a user-defined type --- check to see which
     * scalar it inherits from.
     */

#define _CHK(cls) (PyObject_IsInstance(scalar, \
            (PyObject *)&Py##cls##ArrType_Type))
#define _OBJ(lt) &(((Py##lt##ScalarObject *)scalar)->obval)
#define _IFCASE(cls) if _CHK(cls) return _OBJ(cls)

    if _CHK(Number) {
        if _CHK(Integer) {
            if _CHK(SignedInteger) {
                _IFCASE(Byte);
                _IFCASE(Short);
                _IFCASE(Int);
                _IFCASE(Long);
                _IFCASE(LongLong);
                _IFCASE(Timedelta);
            }
            else {
                /* Unsigned Integer */
                _IFCASE(UByte);
                _IFCASE(UShort);
                _IFCASE(UInt);
                _IFCASE(ULong);
                _IFCASE(ULongLong);
            }
        }
        else {
            /* Inexact */
            if _CHK(Floating) {
                _IFCASE(Half);
                _IFCASE(Float);
                _IFCASE(Double);
                _IFCASE(LongDouble);
            }
            else {
                /*ComplexFloating */
                _IFCASE(CFloat);
                _IFCASE(CDouble);
                _IFCASE(CLongDouble);
            }
        }
    }
    else if (_CHK(Bool)) {
        return _OBJ(Bool);
    }
    else if (_CHK(Datetime)) {
        return _OBJ(Datetime);
    }
    else if (_CHK(Flexible)) {
        if (_CHK(String)) {
            return (void *)PyString_AS_STRING(scalar);
        }
        if (_CHK(Unicode)) {
            return (void *)PyUnicode_AS_DATA(scalar);
        }
        if (_CHK(Void)) {
            return ((PyVoidScalarObject *)scalar)->obval;
        }
    }
    else {
        _IFCASE(Object);
    }


    /*
     * Use the alignment flag to figure out where the data begins
     * after a PyObject_HEAD
     */
    memloc = (npy_intp)scalar;
    memloc += sizeof(PyObject);
    /* now round-up to the nearest alignment value */
    align = descr->alignment;
    if (align > 1) {
        memloc = ((memloc + align - 1)/align)*align;
    }
    return (void *)memloc;
#undef _IFCASE
#undef _OBJ
#undef _CHK
}

/*NUMPY_API
 * return true an object is exactly a numpy scalar
 */
NPY_NO_EXPORT int
PyArray_CheckAnyScalarExact(PyObject * obj)
{
    return is_anyscalar_exact(obj);
}

/*NUMPY_API
 * Convert to c-type
 *
 * no error checking is performed -- ctypeptr must be same type as scalar
 * in case of flexible type, the data is not copied
 * into ctypeptr which is expected to be a pointer to pointer
 */
NPY_NO_EXPORT void
PyArray_ScalarAsCtype(PyObject *scalar, void *ctypeptr)
{
    PyArray_Descr *typecode;
    void *newptr;
    typecode = PyArray_DescrFromScalar(scalar);
    newptr = scalar_value(scalar, typecode);

    if (PyTypeNum_ISEXTENDED(typecode->type_num)) {
        void **ct = (void **)ctypeptr;
        *ct = newptr;
    }
    else {
        memcpy(ctypeptr, newptr, typecode->elsize);
    }
    Py_DECREF(typecode);
    return;
}

/*NUMPY_API
 * Cast Scalar to c-type
 *
 * The output buffer must be large-enough to receive the value
 *  Even for flexible types which is different from ScalarAsCtype
 *  where only a reference for flexible types is returned
 *
 * This may not work right on narrow builds for NumPy unicode scalars.
 */
NPY_NO_EXPORT int
PyArray_CastScalarToCtype(PyObject *scalar, void *ctypeptr,
                          PyArray_Descr *outcode)
{
    PyArray_Descr* descr;
    PyArray_VectorUnaryFunc* castfunc;

    descr = PyArray_DescrFromScalar(scalar);
    castfunc = PyArray_GetCastFunc(descr, outcode->type_num);
    if (castfunc == NULL) {
        return -1;
    }
    if (PyTypeNum_ISEXTENDED(descr->type_num) ||
            PyTypeNum_ISEXTENDED(outcode->type_num)) {
        PyArrayObject *ain, *aout;

        ain = (PyArrayObject *)PyArray_FromScalar(scalar, NULL);
        if (ain == NULL) {
            Py_DECREF(descr);
            return -1;
        }
        aout = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type,
                    outcode,
                    0, NULL,
                    NULL, ctypeptr,
                    NPY_ARRAY_CARRAY, NULL);
        if (aout == NULL) {
            Py_DECREF(ain);
            return -1;
        }
        castfunc(PyArray_DATA(ain), PyArray_DATA(aout), 1, ain, aout);
        Py_DECREF(ain);
        Py_DECREF(aout);
    }
    else {
        castfunc(scalar_value(scalar, descr), ctypeptr, 1, NULL, NULL);
    }
    Py_DECREF(descr);
    return 0;
}

/*NUMPY_API
 * Cast Scalar to c-type
 */
NPY_NO_EXPORT int
PyArray_CastScalarDirect(PyObject *scalar, PyArray_Descr *indescr,
                         void *ctypeptr, int outtype)
{
    PyArray_VectorUnaryFunc* castfunc;
    void *ptr;
    castfunc = PyArray_GetCastFunc(indescr, outtype);
    if (castfunc == NULL) {
        return -1;
    }
    ptr = scalar_value(scalar, indescr);
    castfunc(ptr, ctypeptr, 1, NULL, NULL);
    return 0;
}

/*NUMPY_API
 * Get 0-dim array from scalar
 *
 * 0-dim array from array-scalar object
 * always contains a copy of the data
 * unless outcode is NULL, it is of void type and the referrer does
 * not own it either.
 *
 * steals reference to outcode
 */
NPY_NO_EXPORT PyObject *
PyArray_FromScalar(PyObject *scalar, PyArray_Descr *outcode)
{
    PyArray_Descr *typecode;
    PyArrayObject *r;
    char *memptr;
    PyObject *ret;

    /* convert to 0-dim array of scalar typecode */
    typecode = PyArray_DescrFromScalar(scalar);
    if (typecode == NULL) {
        return NULL;
    }
    if ((typecode->type_num == NPY_VOID) &&
            !(((PyVoidScalarObject *)scalar)->flags & NPY_ARRAY_OWNDATA) &&
            outcode == NULL) {
        return PyArray_NewFromDescrAndBase(
                &PyArray_Type, typecode,
                0, NULL, NULL,
                ((PyVoidScalarObject *)scalar)->obval,
                ((PyVoidScalarObject *)scalar)->flags,
                NULL, (PyObject *)scalar);
    }

    /* Need to INCREF typecode because PyArray_NewFromDescr steals a
     * reference below and we still need to access typecode afterwards. */
    Py_INCREF(typecode);
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
            typecode,
            0, NULL,
            NULL, NULL, 0, NULL);
    if (r==NULL) {
        Py_DECREF(typecode); Py_XDECREF(outcode);
        return NULL;
    }
    if (PyDataType_FLAGCHK(typecode, NPY_USE_SETITEM)) {
        if (typecode->f->setitem(scalar, PyArray_DATA(r), r) < 0) {
            Py_DECREF(typecode); Py_XDECREF(outcode); Py_DECREF(r);
            return NULL;
        }
        goto finish;
    }

    memptr = scalar_value(scalar, typecode);

#ifndef Py_UNICODE_WIDE
    if (typecode->type_num == NPY_UNICODE) {
        PyUCS2Buffer_AsUCS4((Py_UNICODE *)memptr,
                (npy_ucs4 *)PyArray_DATA(r),
                PyUnicode_GET_SIZE(scalar),
                PyArray_ITEMSIZE(r) >> 2);
    }
    else
#endif
    {
        memcpy(PyArray_DATA(r), memptr, PyArray_ITEMSIZE(r));
        if (PyDataType_FLAGCHK(typecode, NPY_ITEM_HASOBJECT)) {
            /* Need to INCREF just the PyObject portion */
            PyArray_Item_INCREF(memptr, typecode);
        }
    }

finish:
    if (outcode == NULL) {
        Py_DECREF(typecode);
        return (PyObject *)r;
    }
    if (PyArray_EquivTypes(outcode, typecode)) {
        if (!PyTypeNum_ISEXTENDED(typecode->type_num)
                || (outcode->elsize == typecode->elsize)) {
            Py_DECREF(typecode); Py_DECREF(outcode);
            return (PyObject *)r;
        }
    }

    /* cast if necessary to desired output typecode */
    ret = PyArray_CastToType((PyArrayObject *)r, outcode, 0);
    Py_DECREF(typecode); Py_DECREF(r);
    return ret;
}

/*NUMPY_API
 * Get an Array Scalar From a Python Object
 *
 * Returns NULL if unsuccessful but error is only set if another error occurred.
 * Currently only Numeric-like object supported.
 */
NPY_NO_EXPORT PyObject *
PyArray_ScalarFromObject(PyObject *object)
{
    PyObject *ret=NULL;
    if (PyArray_IsZeroDim(object)) {
        return PyArray_ToScalar(PyArray_DATA((PyArrayObject *)object),
                                (PyArrayObject *)object);
    }
    /*
     * Booleans in Python are implemented as a subclass of integers,
     * so PyBool_Check must be called before PyInt_Check.
     */
    if (PyBool_Check(object)) {
        if (object == Py_True) {
            PyArrayScalar_RETURN_TRUE;
        }
        else {
            PyArrayScalar_RETURN_FALSE;
        }
    }
    else if (PyInt_Check(object)) {
        ret = PyArrayScalar_New(Long);
        if (ret == NULL) {
            return NULL;
        }
        PyArrayScalar_VAL(ret, Long) = PyInt_AS_LONG(object);
    }
    else if (PyFloat_Check(object)) {
        ret = PyArrayScalar_New(Double);
        if (ret == NULL) {
            return NULL;
        }
        PyArrayScalar_VAL(ret, Double) = PyFloat_AS_DOUBLE(object);
    }
    else if (PyComplex_Check(object)) {
        ret = PyArrayScalar_New(CDouble);
        if (ret == NULL) {
            return NULL;
        }
        PyArrayScalar_VAL(ret, CDouble).real = PyComplex_RealAsDouble(object);
        PyArrayScalar_VAL(ret, CDouble).imag = PyComplex_ImagAsDouble(object);
    }
    else if (PyLong_Check(object)) {
        npy_longlong val;
        val = PyLong_AsLongLong(object);
        if (error_converting(val)) {
            PyErr_Clear();
            return NULL;
        }
        ret = PyArrayScalar_New(LongLong);
        if (ret == NULL) {
            return NULL;
        }
        PyArrayScalar_VAL(ret, LongLong) = val;
    }
    return ret;
}

/*New reference */
/*NUMPY_API
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromTypeObject(PyObject *type)
{
    int typenum;
    PyArray_Descr *new, *conv = NULL;

    /* if it's a builtin type, then use the typenumber */
    typenum = _typenum_fromtypeobj(type,1);
    if (typenum != NPY_NOTYPE) {
        new = PyArray_DescrFromType(typenum);
        return new;
    }

    /* Check the generic types */
    if ((type == (PyObject *) &PyNumberArrType_Type) ||
            (type == (PyObject *) &PyInexactArrType_Type) ||
            (type == (PyObject *) &PyFloatingArrType_Type)) {
        typenum = NPY_DOUBLE;
    }
    else if (type == (PyObject *)&PyComplexFloatingArrType_Type) {
        typenum = NPY_CDOUBLE;
    }
    else if ((type == (PyObject *)&PyIntegerArrType_Type) ||
            (type == (PyObject *)&PySignedIntegerArrType_Type)) {
        typenum = NPY_LONG;
    }
    else if (type == (PyObject *) &PyUnsignedIntegerArrType_Type) {
        typenum = NPY_ULONG;
    }
    else if (type == (PyObject *) &PyCharacterArrType_Type) {
        typenum = NPY_STRING;
    }
    else if ((type == (PyObject *) &PyGenericArrType_Type) ||
            (type == (PyObject *) &PyFlexibleArrType_Type)) {
        typenum = NPY_VOID;
    }

    if (typenum != NPY_NOTYPE) {
        return PyArray_DescrFromType(typenum);
    }

    /*
     * Otherwise --- type is a sub-type of an array scalar
     * not corresponding to a registered data-type object.
     */

    /* Do special thing for VOID sub-types */
    if (PyType_IsSubtype((PyTypeObject *)type, &PyVoidArrType_Type)) {
        new = PyArray_DescrNewFromType(NPY_VOID);
        if (new == NULL) {
            return NULL;
        }
        if (_arraydescr_from_dtype_attr(type, &conv)) {
            if (conv == NULL) {
                Py_DECREF(new);
                return NULL;
            }
            new->fields = conv->fields;
            Py_XINCREF(new->fields);
            new->names = conv->names;
            Py_XINCREF(new->names);
            new->elsize = conv->elsize;
            new->subarray = conv->subarray;
            conv->subarray = NULL;
            Py_DECREF(conv);
        }
        Py_XDECREF(new->typeobj);
        new->typeobj = (PyTypeObject *)type;
        Py_INCREF(type);
        return new;
    }
    return _descr_from_subtype(type);
}

/*NUMPY_API
 * Return the tuple of ordered field names from a dictionary.
 */
NPY_NO_EXPORT PyObject *
PyArray_FieldNames(PyObject *fields)
{
    PyObject *tup;
    PyObject *ret;
    PyObject *_numpy_internal;

    if (!PyDict_Check(fields)) {
        PyErr_SetString(PyExc_TypeError,
                "Fields must be a dictionary");
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    tup = PyObject_CallMethod(_numpy_internal, "_makenames_list", "OO", fields, Py_False);
    Py_DECREF(_numpy_internal);
    if (tup == NULL) {
        return NULL;
    }
    ret = PyTuple_GET_ITEM(tup, 0);
    ret = PySequence_Tuple(ret);
    Py_DECREF(tup);
    return ret;
}

/*NUMPY_API
 * Return descr object from array scalar.
 *
 * New reference
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromScalar(PyObject *sc)
{
    int type_num;
    PyArray_Descr *descr;

    if (PyArray_IsScalar(sc, Void)) {
        descr = ((PyVoidScalarObject *)sc)->descr;
        Py_INCREF(descr);
        return descr;
    }

    if (PyArray_IsScalar(sc, Datetime) || PyArray_IsScalar(sc, Timedelta)) {
        PyArray_DatetimeMetaData *dt_data;

        if (PyArray_IsScalar(sc, Datetime)) {
            descr = PyArray_DescrNewFromType(NPY_DATETIME);
        }
        else {
            /* Timedelta */
            descr = PyArray_DescrNewFromType(NPY_TIMEDELTA);
        }
        if (descr == NULL) {
            return NULL;
        }
        dt_data = &(((PyArray_DatetimeDTypeMetaData *)descr->c_metadata)->meta);
        memcpy(dt_data, &((PyDatetimeScalarObject *)sc)->obmeta,
               sizeof(PyArray_DatetimeMetaData));

        return descr;
    }

    descr = PyArray_DescrFromTypeObject((PyObject *)Py_TYPE(sc));
    if (PyDataType_ISUNSIZED(descr)) {
        PyArray_DESCR_REPLACE(descr);
        type_num = descr->type_num;
        if (type_num == NPY_STRING) {
            descr->elsize = PyString_GET_SIZE(sc);
        }
        else if (type_num == NPY_UNICODE) {
            descr->elsize = PyUnicode_GET_DATA_SIZE(sc);
#ifndef Py_UNICODE_WIDE
            descr->elsize <<= 1;
#endif
        }
        else {
            PyArray_Descr *dtype;
            dtype = (PyArray_Descr *)PyObject_GetAttrString(sc, "dtype");
            if (dtype != NULL) {
                descr->elsize = dtype->elsize;
                descr->fields = dtype->fields;
                Py_XINCREF(dtype->fields);
                descr->names = dtype->names;
                Py_XINCREF(dtype->names);
                Py_DECREF(dtype);
            }
            PyErr_Clear();
        }
    }
    return descr;
}

/*NUMPY_API
 * Get a typeobject from a type-number -- can return NULL.
 *
 * New reference
 */
NPY_NO_EXPORT PyObject *
PyArray_TypeObjectFromType(int type)
{
    PyArray_Descr *descr;
    PyObject *obj;

    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        return NULL;
    }
    obj = (PyObject *)descr->typeobj;
    Py_XINCREF(obj);
    Py_DECREF(descr);
    return obj;
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
    if (type_num == NPY_BOOL) {
        PyArrayScalar_RETURN_BOOL_FROM_LONG(*(npy_bool*)data);
    }
    else if (PyDataType_FLAGCHK(descr, NPY_USE_GETITEM)) {
        return descr->f->getitem(data, base);
    }
    itemsize = descr->elsize;
    copyswap = descr->f->copyswap;
    type = descr->typeobj;
    swap = !PyArray_ISNBO(descr->byteorder);
    if (PyTypeNum_ISSTRING(type_num)) {
        /* Eliminate NULL bytes */
        char *dptr = data;

        dptr += itemsize - 1;
        while(itemsize && *dptr-- == 0) {
            itemsize--;
        }
        if (type_num == NPY_UNICODE && itemsize) {
            /*
             * make sure itemsize is a multiple of 4
             * so round up to nearest multiple
             */
            itemsize = (((itemsize - 1) >> 2) + 1) << 2;
        }
    }
#if PY_VERSION_HEX >= 0x03030000
    if (type_num == NPY_UNICODE) {
        PyObject *u, *args;
        int byteorder;

#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
        byteorder = -1;
#elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN
        byteorder = +1;
#else
        #error Endianness undefined ?
#endif
        if (swap) byteorder *= -1;

        u = PyUnicode_DecodeUTF32(data, itemsize, NULL, &byteorder);
        if (u == NULL) {
            return NULL;
        }
        args = Py_BuildValue("(O)", u);
        if (args == NULL) {
            Py_DECREF(u);
            return NULL;
        }
        obj = type->tp_new(type, args, NULL);
        Py_DECREF(u);
        Py_DECREF(args);
        return obj;
    }
#endif
    if (type->tp_itemsize != 0) {
        /* String type */
        obj = type->tp_alloc(type, itemsize);
    }
    else {
        obj = type->tp_alloc(type, 0);
    }
    if (obj == NULL) {
        return NULL;
    }
    if (PyTypeNum_ISDATETIME(type_num)) {
        /*
         * We need to copy the resolution information over to the scalar
         * Get the void * from the metadata dictionary
         */
        PyArray_DatetimeMetaData *dt_data;

        dt_data = &(((PyArray_DatetimeDTypeMetaData *)descr->c_metadata)->meta);
        memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
               sizeof(PyArray_DatetimeMetaData));
    }
    if (PyTypeNum_ISFLEXIBLE(type_num)) {
        if (type_num == NPY_STRING) {
            destptr = PyString_AS_STRING(obj);
            ((PyStringObject *)obj)->ob_shash = -1;
#if !defined(NPY_PY3K)
            ((PyStringObject *)obj)->ob_sstate = SSTATE_NOT_INTERNED;
#endif
            memcpy(destptr, data, itemsize);
            return obj;
        }
#if PY_VERSION_HEX < 0x03030000
        else if (type_num == NPY_UNICODE) {
            /* tp_alloc inherited from Python PyBaseObject_Type */
            PyUnicodeObject *uni = (PyUnicodeObject*)obj;
            size_t length = itemsize >> 2;
            Py_UNICODE *dst;
#ifndef Py_UNICODE_WIDE
            char *buffer;
            Py_UNICODE *tmp;
            int alloc = 0;

            length *= 2;
#endif
            /* Set uni->str so that object can be deallocated on failure */
            uni->str = NULL;
            uni->defenc = NULL;
            uni->hash = -1;
            dst = PyObject_MALLOC(sizeof(Py_UNICODE) * (length + 1));
            if (dst == NULL) {
                Py_DECREF(obj);
                PyErr_NoMemory();
                return NULL;
            }
#ifdef Py_UNICODE_WIDE
            memcpy(dst, data, itemsize);
            if (swap) {
                byte_swap_vector(dst, length, 4);
            }
            uni->str = dst;
            uni->str[length] = 0;
            uni->length = length;
#else
            /* need aligned data buffer */
            if ((swap) || ((((npy_intp)data) % descr->alignment) != 0)) {
                buffer = malloc(itemsize);
                if (buffer == NULL) {
                    PyObject_FREE(dst);
                    Py_DECREF(obj);
                    PyErr_NoMemory();
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

            /*
             * Allocated enough for 2-characters per itemsize.
             * Now convert from the data-buffer
             */
            length = PyUCS2Buffer_FromUCS4(dst,
                    (npy_ucs4 *)buffer, itemsize >> 2);
            if (alloc) {
                free(buffer);
            }
            /* Resize the unicode result */
            tmp = PyObject_REALLOC(dst, sizeof(Py_UNICODE)*(length + 1));
            if (tmp == NULL) {
                PyObject_FREE(dst);
                Py_DECREF(obj);
                return NULL;
            }
            uni->str = tmp;
            uni->str[length] = 0;
            uni->length = length;
#endif
            return obj;
        }
#endif /* PY_VERSION_HEX < 0x03030000 */
        else {
            PyVoidScalarObject *vobj = (PyVoidScalarObject *)obj;
            vobj->base = NULL;
            vobj->descr = descr;
            Py_INCREF(descr);
            vobj->obval = NULL;
            Py_SIZE(vobj) = itemsize;
            vobj->flags = NPY_ARRAY_CARRAY | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA;
            swap = 0;
            if (PyDataType_HASFIELDS(descr)) {
                if (base) {
                    Py_INCREF(base);
                    vobj->base = base;
                    vobj->flags = PyArray_FLAGS((PyArrayObject *)base);
                    vobj->flags &= ~NPY_ARRAY_OWNDATA;
                    vobj->obval = data;
                    return obj;
                }
            }
            if (itemsize == 0) {
                return obj;
            }
            destptr = PyDataMem_NEW(itemsize);
            if (destptr == NULL) {
                Py_DECREF(obj);
                return PyErr_NoMemory();
            }
            vobj->obval = destptr;

            /*
             * No base available for copyswp and no swap required.
             * Copy data directly into dest.
             */
            if (base == NULL) {
                memcpy(destptr, data, itemsize);
                return obj;
            }
        }
    }
    else {
        destptr = scalar_value(obj, descr);
    }
    /* copyswap for OBJECT increments the reference count */
    copyswap(destptr, data, swap, base);
    return obj;
}

/* Return Array Scalar if 0-d array object is encountered */

/*NUMPY_API
 *
 * Return either an array or the appropriate Python object if the array
 * is 0d and matches a Python type.
 * steals reference to mp
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
    if (PyArray_NDIM(mp) == 0) {
        PyObject *ret;
        ret = PyArray_ToScalar(PyArray_DATA(mp), mp);
        Py_DECREF(mp);
        return ret;
    }
    else {
        return (PyObject *)mp;
    }
}

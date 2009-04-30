#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "arrayobject.h"

static PyArray_Descr *
_descr_from_subtype(PyObject *type)
{
    PyObject *mro;
    mro = ((PyTypeObject *)type)->tp_mro;
    if (PyTuple_GET_SIZE(mro) < 2) {
        return PyArray_DescrFromType(PyArray_OBJECT);
    }
    return PyArray_DescrFromTypeObject(PyTuple_GET_ITEM(mro, 1));
}

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr)
{
    int type_num;
    int align;
    intp memloc;
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
        CASE(FLOAT, Float);
        CASE(DOUBLE, Double);
        CASE(LONGDOUBLE, LongDouble);
        CASE(CFLOAT, CFloat);
        CASE(CDOUBLE, CDouble);
        CASE(CLONGDOUBLE, CLongDouble);
        CASE(OBJECT, Object);
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
    memloc = (intp)scalar;
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
                    CARRAY, NULL);
        if (aout == NULL) {
            Py_DECREF(ain);
            return -1;
        }
        castfunc(ain->data, aout->data, 1, ain, aout);
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
    PyObject *r;
    char *memptr;
    PyObject *ret;

    /* convert to 0-dim array of scalar typecode */
    typecode = PyArray_DescrFromScalar(scalar);
    if ((typecode->type_num == PyArray_VOID) &&
            !(((PyVoidScalarObject *)scalar)->flags & OWNDATA) &&
            outcode == NULL) {
        r = PyArray_NewFromDescr(&PyArray_Type,
                typecode,
                0, NULL, NULL,
                ((PyVoidScalarObject *)scalar)->obval,
                ((PyVoidScalarObject *)scalar)->flags,
                NULL);
        PyArray_BASE(r) = (PyObject *)scalar;
        Py_INCREF(scalar);
        return r;
    }
    r = PyArray_NewFromDescr(&PyArray_Type,
            typecode,
            0, NULL,
            NULL, NULL, 0, NULL);
    if (r==NULL) {
        Py_XDECREF(outcode);
        return NULL;
    }
    if (PyDataType_FLAGCHK(typecode, NPY_USE_SETITEM)) {
        if (typecode->f->setitem(scalar, PyArray_DATA(r), r) < 0) {
            Py_XDECREF(outcode); Py_DECREF(r);
            return NULL;
        }
        goto finish;
    }

    memptr = scalar_value(scalar, typecode);

#ifndef Py_UNICODE_WIDE
    if (typecode->type_num == PyArray_UNICODE) {
        PyUCS2Buffer_AsUCS4((Py_UNICODE *)memptr,
                (PyArray_UCS4 *)PyArray_DATA(r),
                PyUnicode_GET_SIZE(scalar),
                PyArray_ITEMSIZE(r) >> 2);
    }
    else
#endif
    {
        memcpy(PyArray_DATA(r), memptr, PyArray_ITEMSIZE(r));
        if (PyDataType_FLAGCHK(typecode, NPY_ITEM_HASOBJECT)) {
            Py_INCREF(*((PyObject **)memptr));
        }
    }

finish:
    if (outcode == NULL) {
        return r;
    }
    if (outcode->type_num == typecode->type_num) {
        if (!PyTypeNum_ISEXTENDED(typecode->type_num) ||
                (outcode->elsize == typecode->elsize))
            return r;
    }

    /* cast if necessary to desired output typecode */
    ret = PyArray_CastToType((PyArrayObject *)r, outcode, 0);
    Py_DECREF(r);
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
        return PyArray_ToScalar(PyArray_DATA(object), object);
    }
    if (PyInt_Check(object)) {
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
        PyArrayScalar_VAL(ret, CDouble).real =
                ((PyComplexObject *)object)->cval.real;
        PyArrayScalar_VAL(ret, CDouble).imag =
                ((PyComplexObject *)object)->cval.imag;
    }
    else if (PyLong_Check(object)) {
        longlong val;
        val = PyLong_AsLongLong(object);
        if (val==-1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NULL;
        }
        ret = PyArrayScalar_New(LongLong);
        if (ret == NULL) {
            return NULL;
        }
        PyArrayScalar_VAL(ret, LongLong) = val;
    }
    else if (PyBool_Check(object)) {
        if (object == Py_True) {
            PyArrayScalar_RETURN_TRUE;
        }
        else {
            PyArrayScalar_RETURN_FALSE;
        }
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
    PyArray_Descr *new, *conv=NULL;

    /* if it's a builtin type, then use the typenumber */
    typenum = _typenum_fromtypeobj(type,1);
    if (typenum != PyArray_NOTYPE) {
        new = PyArray_DescrFromType(typenum);
        return new;
    }

    /* Check the generic types */
    if ((type == (PyObject *) &PyNumberArrType_Type) ||
            (type == (PyObject *) &PyInexactArrType_Type) ||
            (type == (PyObject *) &PyFloatingArrType_Type)) {
        typenum = PyArray_DOUBLE;
    }
    else if (type == (PyObject *)&PyComplexFloatingArrType_Type) {
        typenum = PyArray_CDOUBLE;
    }
    else if ((type == (PyObject *)&PyIntegerArrType_Type) ||
            (type == (PyObject *)&PySignedIntegerArrType_Type)) {
        typenum = PyArray_LONG;
    }
    else if (type == (PyObject *) &PyUnsignedIntegerArrType_Type) {
        typenum = PyArray_ULONG;
    }
    else if (type == (PyObject *) &PyCharacterArrType_Type) {
        typenum = PyArray_STRING;
    }
    else if ((type == (PyObject *) &PyGenericArrType_Type) ||
            (type == (PyObject *) &PyFlexibleArrType_Type)) {
        typenum = PyArray_VOID;
    }

    if (typenum != PyArray_NOTYPE) {
        return PyArray_DescrFromType(typenum);
    }

    /*
     * Otherwise --- type is a sub-type of an array scalar
     * not corresponding to a registered data-type object.
     */

    /* Do special thing for VOID sub-types */
    if (PyType_IsSubtype((PyTypeObject *)type, &PyVoidArrType_Type)) {
        new = PyArray_DescrNewFromType(PyArray_VOID);
        conv = _arraydescr_fromobj(type);
        if (conv) {
            new->fields = conv->fields;
            Py_INCREF(new->fields);
            new->names = conv->names;
            Py_INCREF(new->names);
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
    tup = PyObject_CallMethod(_numpy_internal, "_makenames_list", "O", fields);
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
    descr = PyArray_DescrFromTypeObject((PyObject *)sc->ob_type);
    if (descr->elsize == 0) {
        PyArray_DESCR_REPLACE(descr);
        type_num = descr->type_num;
        if (type_num == PyArray_STRING) {
            descr->elsize = PyString_GET_SIZE(sc);
        }
        else if (type_num == PyArray_UNICODE) {
            descr->elsize = PyUnicode_GET_DATA_SIZE(sc);
#ifndef Py_UNICODE_WIDE
            descr->elsize <<= 1;
#endif
        }
        else {
            descr->elsize =
                ((PyVoidScalarObject *)sc)->ob_size;
            descr->fields = PyObject_GetAttrString(sc, "fields");
            if (!descr->fields || !PyDict_Check(descr->fields) ||
                    (descr->fields == Py_None)) {
                Py_XDECREF(descr->fields);
                descr->fields = NULL;
            }
            if (descr->fields) {
                descr->names = PyArray_FieldNames(descr->fields);
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

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"



#include "array_coercion.h"
#include "ctors.h"
#include "descriptor.h"
#include "dtypemeta.h"
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
    uintptr_t memloc;
    if (descr == NULL) {
        descr = PyArray_DescrFromScalar(scalar);
        type_num = descr->type_num;
        Py_DECREF(descr);
    }
    else {
        type_num = descr->type_num;
    }
    switch (type_num) {
#define CASE(ut,lt) case NPY_##ut: return &PyArrayScalar_VAL(scalar, lt)
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
            return (void *)PyBytes_AsString(scalar);
        case NPY_UNICODE:
            /* lazy initialization, to reduce the memory used by string scalars */
            if (PyArrayScalar_VAL(scalar, Unicode) == NULL) {
                Py_UCS4 *raw_data = PyUnicode_AsUCS4Copy(scalar);
                if (raw_data == NULL) {
                    return NULL;
                }
                PyArrayScalar_VAL(scalar, Unicode) = raw_data;
                return (void *)raw_data;
            }
            return PyArrayScalar_VAL(scalar, Unicode);
        case NPY_VOID:
            /* Note: no & needed here, so can't use CASE */
            return PyArrayScalar_VAL(scalar, Void);
    }

    /*
     * Must be a user defined type with an associated (registered) dtype.
     * Thus, it cannot be flexible (user dtypes cannot be), so we can (and
     * pretty much have no choice but to) assume the below logic always works.
     * I.e. this assumes that the logic would also works for most of our types.
     */

    /*
     * Use the alignment flag to figure out where the data begins
     * after a PyObject_HEAD
     */
    memloc = (uintptr_t)scalar;
    memloc += sizeof(PyObject);
    /* now round-up to the nearest alignment value */
    align = descr->alignment;
    if (align > 1) {
        memloc = ((memloc + align - 1)/align)*align;
    }
    return (void *)memloc;
}

/*NUMPY_API
 * return 1 if an object is exactly a numpy scalar
 */
NPY_NO_EXPORT int
PyArray_CheckAnyScalarExact(PyObject * obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "obj is NULL in PyArray_CheckAnyScalarExact");
        return 0;
    }

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
 * The output buffer must be large-enough to receive the value, this function
 * should only be used for subclasses of `np.generic`, we can only guarantee
 * it works for NumPy builtins.
 */
NPY_NO_EXPORT int
PyArray_CastScalarToCtype(PyObject *scalar, void *ctypeptr,
                          PyArray_Descr *outcode)
{
    PyArray_Descr* descr;

    descr = PyArray_DescrFromScalar(scalar);
    if (descr == NULL) {
        return -1;
    }
    void *src = scalar_value(scalar, descr);
    if (src == NULL) {
        Py_DECREF(descr);
        return -1;
    }

    int res = npy_cast_raw_scalar_item(descr, src, outcode, ctypeptr);
    Py_DECREF(descr);
    return res;
}

/*NUMPY_API
 * Cast Scalar to c-type
 */
NPY_NO_EXPORT int
PyArray_CastScalarDirect(PyObject *scalar, PyArray_Descr *indescr,
                         void *ctypeptr, int outtype)
{
    PyArray_Descr *out_dt = PyArray_DescrFromType(outtype);
    if (out_dt == NULL) {
        return -1;
    }
    void *src = scalar_value(scalar, indescr);
    if (src == NULL) {
        Py_DECREF(out_dt);
        return -1;
    }

    int res = npy_cast_raw_scalar_item(indescr, src, out_dt, ctypeptr);
    Py_DECREF(out_dt);
    return res;
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
    /* convert to 0-dim array of scalar typecode */
    PyArray_Descr *typecode = PyArray_DescrFromScalar(scalar);
    if (typecode == NULL) {
        Py_XDECREF(outcode);
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

    PyArrayObject *r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
            typecode,
            0, NULL,
            NULL, NULL, 0, NULL);
    if (r == NULL) {
        Py_XDECREF(outcode);
        return NULL;
    }
    /* the dtype used by the array may be different to the one requested */
    typecode = PyArray_DESCR(r);
    if (PyDataType_FLAGCHK(typecode, NPY_USE_SETITEM)) {
        if (PyDataType_GetArrFuncs(typecode)->setitem(scalar, PyArray_DATA(r), r) < 0) {
            Py_DECREF(r);
            Py_XDECREF(outcode);
            return NULL;
        }
    }
    else {
        char *memptr = scalar_value(scalar, typecode);

        memcpy(PyArray_DATA(r), memptr, PyArray_ITEMSIZE(r));
        if (PyDataType_FLAGCHK(typecode, NPY_ITEM_HASOBJECT)) {
            /* Need to INCREF just the PyObject portion */
            PyArray_Item_INCREF(memptr, typecode);
        }
    }

    if (outcode == NULL) {
        return (PyObject *)r;
    }
    if (PyArray_EquivTypes(outcode, typecode)) {
        if (!PyTypeNum_ISEXTENDED(typecode->type_num)
                || (outcode->elsize == typecode->elsize)) {
            /*
             * Since the type is equivalent, and we haven't handed the array
             * to anyone yet, let's fix the dtype to be what was requested,
             * even if it is equivalent to what was passed in.
             */
            Py_SETREF(((PyArrayObject_fields *)r)->descr, outcode);

            return (PyObject *)r;
        }
    }

    /* cast if necessary to desired output typecode */
    PyObject *ret = PyArray_CastToType(r, outcode, 0);
    Py_DECREF(r);
    return ret;
}


/*New reference */
/*NUMPY_API
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromTypeObject(PyObject *type)
{
    /* if it's a builtin type, then use the typenumber */
    int typenum = _typenum_fromtypeobj(type,1);
    if (typenum != NPY_NOTYPE) {
        return PyArray_DescrFromType(typenum);
    }

    /* Check the generic types, was deprecated in 1.19 and removed for 2.3 */
    if ((type == (PyObject *) &PyNumberArrType_Type) ||
            (type == (PyObject *) &PyInexactArrType_Type) ||
            (type == (PyObject *) &PyFloatingArrType_Type)) {
        PyErr_SetString(PyExc_TypeError,
            "Converting `np.inexact` or `np.floating` to "
                      "a dtype not allowed");
        return NULL;
    }
    else if (type == (PyObject *)&PyComplexFloatingArrType_Type) {
        PyErr_SetString(PyExc_TypeError,
            "Converting `np.complex` to a dtype is not allowed.");
        return NULL;
    }
    else if ((type == (PyObject *)&PyIntegerArrType_Type) ||
            (type == (PyObject *)&PySignedIntegerArrType_Type)) {
        PyErr_SetString(PyExc_TypeError,
            "Converting 'np.integer' or 'np.signedinteger' to "
                      "a dtype is not allowed");
        return NULL;
    }
    else if (type == (PyObject *) &PyUnsignedIntegerArrType_Type) {
        PyErr_SetString(PyExc_TypeError,
            "Converting `np.unsignedinteger` to a dtype is not allowed");
        return NULL;
    }
    else if (type == (PyObject *) &PyCharacterArrType_Type) {
        PyErr_SetString(PyExc_TypeError,
            "Converting `np.character` to a dtype is not allowed");
        return NULL;
    }
    else if ((type == (PyObject *) &PyGenericArrType_Type) ||
            (type == (PyObject *) &PyFlexibleArrType_Type)) {
        PyErr_SetString(PyExc_TypeError,
            "Converting `np.generic` to a dtype is not allowed.");
        return NULL;
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
        _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr  *)PyArray_DescrNewFromType(NPY_VOID);
        if (new == NULL) {
            return NULL;
        }
        _PyArray_LegacyDescr *conv = (_PyArray_LegacyDescr *)(
                _arraydescr_try_convert_from_dtype_attr(type));
        if (conv == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        if ((PyObject *)conv != Py_NotImplemented && PyDataType_ISLEGACY(conv)) {
            new->fields = conv->fields;
            Py_XINCREF(new->fields);
            new->names = conv->names;
            Py_XINCREF(new->names);
            new->elsize = conv->elsize;
            new->subarray = conv->subarray;
            conv->subarray = NULL;
        }
        Py_DECREF(conv);
        Py_XDECREF(new->typeobj);
        new->typeobj = (PyTypeObject *)type;
        Py_INCREF(type);
        return (PyArray_Descr *)new;
    }

    PyObject *DType = PyArray_DiscoverDTypeFromScalarType((PyTypeObject *)type);
    if (DType != NULL) {
        return PyArray_GetDefaultDescr((PyArray_DTypeMeta *)DType);
    }

    return _descr_from_subtype(type);
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
        descr = (PyArray_Descr *)((PyVoidScalarObject *)sc)->descr;
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
        dt_data = &(((PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)descr)->c_metadata)->meta);
        memcpy(dt_data, &((PyDatetimeScalarObject *)sc)->obmeta,
               sizeof(PyArray_DatetimeMetaData));

        return descr;
    }

    descr = PyArray_DescrFromTypeObject((PyObject *)Py_TYPE(sc));
    if (descr == NULL) {
        return NULL;
    }
    if (PyDataType_ISLEGACY(descr) && PyDataType_ISUNSIZED(descr)) {
        PyArray_DESCR_REPLACE(descr);
        if (descr == NULL) {
            return NULL;
        }
        type_num = descr->type_num;
        if (type_num == NPY_STRING) {
            descr->elsize = PyBytes_GET_SIZE(sc);
        }
        else if (type_num == NPY_UNICODE) {
            descr->elsize = PyUnicode_GET_LENGTH(sc) * 4;
        }
        else {
            _PyArray_LegacyDescr *ldescr = (_PyArray_LegacyDescr *)descr;
            PyArray_Descr *dtype;
            dtype = (PyArray_Descr *)PyObject_GetAttrString(sc, "dtype");
            if (dtype != NULL) {
                descr->elsize = dtype->elsize;
                ldescr->fields = PyDataType_FIELDS(dtype);
                Py_XINCREF(ldescr->fields);
                ldescr->names = PyDataType_NAMES(dtype);
                Py_XINCREF(ldescr->names);
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
        return PyDataType_GetArrFuncs(descr)->getitem(data, base);
    }
    itemsize = descr->elsize;
    copyswap = PyDataType_GetArrFuncs(descr)->copyswap;
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
    if (type_num == NPY_UNICODE) {
        /* we need the full string length here, else copyswap will write too
           many bytes */
        void *buff = PyArray_malloc(descr->elsize);
        if (buff == NULL) {
            return PyErr_NoMemory();
        }
        /* copyswap needs an array object, but only actually cares about the
         * dtype
         */
        PyArrayObject_fields dummy_arr;
        if (base == NULL) {
            dummy_arr.descr = descr;
            base = (PyObject *)&dummy_arr;
        }
        copyswap(buff, data, swap, base);

        /* truncation occurs here */
        PyObject *u = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buff, itemsize / 4);
        PyArray_free(buff);
        if (u == NULL) {
            return NULL;
        }

        PyObject *args = Py_BuildValue("(O)", u);
        if (args == NULL) {
            Py_DECREF(u);
            return NULL;
        }
        obj = type->tp_new(type, args, NULL);
        Py_DECREF(u);
        Py_DECREF(args);
        return obj;
    }
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

        dt_data = &(((PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)descr)->c_metadata)->meta);
        memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
               sizeof(PyArray_DatetimeMetaData));
    }
    if (PyTypeNum_ISFLEXIBLE(type_num)) {
        if (type_num == NPY_STRING) {
            destptr = PyBytes_AS_STRING(obj);
            memcpy(destptr, data, itemsize);
            return obj;
        }
        else {
            PyVoidScalarObject *vobj = (PyVoidScalarObject *)obj;
            vobj->base = NULL;
            vobj->descr = (_PyArray_LegacyDescr *)descr;
            Py_INCREF(descr);
            vobj->obval = NULL;
            Py_SET_SIZE(vobj, itemsize);
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

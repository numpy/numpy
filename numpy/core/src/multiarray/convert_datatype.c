#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "common.h"
#include "scalartypes.h"
#include "mapping.h"

#include "convert_datatype.h"

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

    out = PyArray_NewFromDescr(Py_TYPE(mp), at,
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
            if (NpyCapsule_Check(cobj)) {
                castfunc = NpyCapsule_AsVoidPtr(cobj);
            }
        }
    }
    if (PyTypeNum_ISCOMPLEX(descr->type_num) &&
        !PyTypeNum_ISCOMPLEX(type_num) &&
        PyTypeNum_ISNUMBER(type_num) &&
        !PyTypeNum_ISBOOL(type_num)) {
        PyObject *cls = NULL, *obj = NULL;
        int ret;
        obj = PyImport_ImportModule("numpy.core");
        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            Py_DECREF(obj);
        }
#if PY_VERSION_HEX >= 0x02050000
        ret = PyErr_WarnEx(cls,
                           "Casting complex values to real discards the imaginary "
                           "part", 1);
#else
        ret = PyErr_Warn(cls,
                         "Casting complex values to real discards the imaginary "
                         "part");
#endif
        Py_XDECREF(cls);
        if (ret < 0) {
            return NULL;
	}
    }
    if (castfunc) {
        return castfunc;
    }

    PyErr_SetString(PyExc_ValueError, "No cast function available.");
    return NULL;
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
    /* CopyInto handles the casting now */
    return PyArray_CopyInto(out, mp);
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
NPY_NO_EXPORT int
PyArray_CastAnyTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyAnyInto handles the casting now */
    return PyArray_CopyAnyInto(out, mp);
}

/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    PyArray_Descr *from;

    /* Fast table lookup for small type numbers */
    if ((unsigned int)fromtype < NPY_NTYPES && (unsigned int)totype < NPY_NTYPES) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    /* Identity */
    if (fromtype == totype) {
        return 1;
    }
    /* Special-cases for some types */
    switch (fromtype) {
        case PyArray_DATETIME:
        case PyArray_TIMEDELTA:
        case PyArray_OBJECT:
        case PyArray_VOID:
            return 0;
        case PyArray_BOOL:
            return 1;
    }
    switch (totype) {
        case PyArray_BOOL:
        case PyArray_DATETIME:
        case PyArray_TIMEDELTA:
            return 0;
        case PyArray_OBJECT:
        case PyArray_VOID:
            return 1;
    }

    from = PyArray_DescrFromType(fromtype);
    /*
     * cancastto is a PyArray_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    if (from->f->cancastto) {
        int *curtype = from->f->cancastto;

        while (*curtype != PyArray_NOTYPE) {
            if (*curtype++ == totype) {
                return 1;
            }
        }
    }
    return 0;
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

static int
_check_object_rec(PyArray_Descr *descr)
{
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

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

/* Raises error when len(op) == 0 */

/*NUMPY_API*/
NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n, allscalars = 0;
    PyArrayObject **mps = NULL;
    PyObject *otmp;
    PyArray_Descr *intype = NULL, *stype = NULL;
    PyArray_Descr *newtype = NULL;
    NPY_SCALARKIND scalarkind = NPY_NOSCALAR, intypekind = NPY_NOSCALAR;

    *retn = n = PySequence_Length(op);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        return (void*)PyErr_NoMemory();
    }

    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            mps[i] = (PyArrayObject *) array_big_item((PyArrayObject *)op, i);
        }
        if (!PyArray_ISCARRAY(op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        return mps;
    }

    for (i = 0; i < n; i++) {
        otmp = PySequence_GetItem(op, i);
        if (!PyArray_CheckAnyScalar(otmp)) {
            newtype = PyArray_DescrFromObject(otmp, intype);
            Py_XDECREF(intype);
            intype = newtype;
            mps[i] = NULL;
            intypekind = PyArray_ScalarKind(intype->type_num, NULL);
        }
        else {
            newtype = PyArray_DescrFromObject(otmp, stype);
            Py_XDECREF(stype);
            stype = newtype;
            scalarkind = PyArray_ScalarKind(newtype->type_num, NULL);
            mps[i] = (PyArrayObject *)Py_None;
            Py_INCREF(Py_None);
        }
        Py_XDECREF(otmp);
    }
    if (intype==NULL) {
        /* all scalars */
        allscalars = 1;
        intype = stype;
        Py_INCREF(intype);
        for (i = 0; i < n; i++) {
            Py_XDECREF(mps[i]);
            mps[i] = NULL;
        }
    }
    else if ((stype != NULL) && (intypekind != scalarkind)) {
        /*
         * we need to upconvert to type that
         * handles both intype and stype
         * also don't forcecast the scalars.
         */
        if (!PyArray_CanCoerceScalar(stype->type_num,
                                     intype->type_num,
                                     scalarkind)) {
            newtype = _array_small_type(intype, stype);
            Py_XDECREF(intype);
            intype = newtype;
        }
        for (i = 0; i < n; i++) {
            Py_XDECREF(mps[i]);
            mps[i] = NULL;
        }
    }


    /* Make sure all arrays are actual array objects. */
    for (i = 0; i < n; i++) {
        int flags = CARRAY;

        if ((otmp = PySequence_GetItem(op, i)) == NULL) {
            goto fail;
        }
        if (!allscalars && ((PyObject *)(mps[i]) == Py_None)) {
            /* forcecast scalars */
            flags |= FORCECAST;
            Py_DECREF(Py_None);
        }
        Py_INCREF(intype);
        mps[i] = (PyArrayObject*)
            PyArray_FromAny(otmp, intype, 0, 0, flags, NULL);
        Py_DECREF(otmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(intype);
    Py_XDECREF(stype);
    return mps;

 fail:
    Py_XDECREF(intype);
    Py_XDECREF(stype);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}


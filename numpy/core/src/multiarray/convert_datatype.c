#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

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
    intp mpsize = PyArray_SIZE(mp);
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
    npy_intp mpsize = PyArray_SIZE(mp);

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
    if (fromtype == PyArray_DATETIME || fromtype == PyArray_TIMEDELTA ||
            totype == PyArray_DATETIME || totype == PyArray_TIMEDELTA) {
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


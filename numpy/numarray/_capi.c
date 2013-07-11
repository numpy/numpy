#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _libnumarray_MODULE
#include "include/numpy/libnumarray.h"
#include "numpy/npy_3kcompat.h"
#include <float.h>

#if (defined(__unix__) || defined(unix)) && !defined(USG)
#include <sys/param.h>
#endif

#if defined(__GLIBC__) || defined(__APPLE__) || defined(__MINGW32__) || (defined(__FreeBSD__) && (__FreeBSD_version >= 502114))
#include <fenv.h>
#elif defined(__CYGWIN__)
#include "numpy/fenv/fenv.h"
#include "numpy/fenv/fenv.c"
#endif

static PyObject *pCfuncClass;
static PyTypeObject CfuncType;
static PyObject *pHandleErrorFunc;

static int
deferred_libnumarray_init(void)
{
static int initialized=0;

    if (initialized) return 0;

    pCfuncClass = (PyObject *) &CfuncType;
    Py_INCREF(pCfuncClass);

    pHandleErrorFunc =
        NA_initModuleGlobal("numpy.numarray.util", "handleError");

    if (!pHandleErrorFunc) goto _fail;


    /* _exit: */
    initialized = 1;
    return 0;

_fail:
    initialized = 0;
    return -1;
}



/**********************************************************************/
/*  Buffer Utility Functions                                          */
/**********************************************************************/

static PyObject *
getBuffer( PyObject *obj)
{
    if (!obj) return PyErr_Format(PyExc_RuntimeError,
            "NULL object passed to getBuffer()");
    if (((PyObject*)obj)->ob_type->tp_as_buffer == NULL) {
        return PyObject_CallMethod(obj, "__buffer__", NULL);
    } else {
        Py_INCREF(obj);  /* Since CallMethod returns a new object when it
                            succeeds, We'll need to DECREF later to free it.
                            INCREF ordinary buffers here so we don't have to
                            remember where the buffer came from at DECREF time.
                            */
        return obj;
    }
}

/* Either it defines the buffer API, or it is an instance which returns
   a buffer when obj.__buffer__() is called */
static int
isBuffer (PyObject *obj)
{
    PyObject *buf = getBuffer(obj);
    int ans = 0;
    if (buf) {
        ans = buf->ob_type->tp_as_buffer != NULL;
        Py_DECREF(buf);
    } else {
        PyErr_Clear();
    }
    return ans;
}

/**********************************************************************/

static int
getWriteBufferDataPtr(PyObject *buffobj, void **buff)
{
#if defined(NPY_PY3K)
    /* FIXME: XXX - needs implementation */
    PyErr_SetString(PyExc_RuntimeError,
                    "XXX: getWriteBufferDataPtr is not implemented");
    return -1;
#else
    int rval = -1;
    PyObject *buff2;
    if ((buff2 = getBuffer(buffobj)))
    {
        if (buff2->ob_type->tp_as_buffer->bf_getwritebuffer)
            rval = buff2->ob_type->tp_as_buffer->bf_getwritebuffer(buff2,
                    0, buff);
        Py_DECREF(buff2);
    }
    return rval;
#endif
}

/**********************************************************************/

static int
isBufferWriteable (PyObject *buffobj)
{
    void *ptr;
    int rval = -1;
    rval = getWriteBufferDataPtr(buffobj, &ptr);
    if (rval == -1)
        PyErr_Clear(); /* Since we're just "testing", it's not really an error */
    return rval != -1;
}

/**********************************************************************/

static int
getReadBufferDataPtr(PyObject *buffobj, void **buff)
{
#if defined(NPY_PY3K)
    /* FIXME: XXX - needs implementation */
    PyErr_SetString(PyExc_RuntimeError,
                    "XXX: getWriteBufferDataPtr is not implemented");
    return -1;
#else
    int rval = -1;
    PyObject *buff2;
    if ((buff2 = getBuffer(buffobj))) {
        if (buff2->ob_type->tp_as_buffer->bf_getreadbuffer)
            rval = buff2->ob_type->tp_as_buffer->bf_getreadbuffer(buff2,
                    0, buff);
        Py_DECREF(buff2);
    }
    return rval;
#endif
}

/**********************************************************************/

static int
getBufferSize(PyObject *buffobj)
{
#if defined(NPY_PY3K)
    /* FIXME: XXX - needs implementation */
    PyErr_SetString(PyExc_RuntimeError,
                    "XXX: getWriteBufferDataPtr is not implemented");
    return -1;
#else
    Py_ssize_t size=0;
    PyObject *buff2;
    if ((buff2 = getBuffer(buffobj)))
    {
        (void) buff2->ob_type->tp_as_buffer->bf_getsegcount(buff2, &size);
        Py_DECREF(buff2);
    }
    else
        size = -1;
    return size;
#endif
}


static double numarray_zero = 0.0;

static double raiseDivByZero(void)
{
    return 1.0/numarray_zero;
}

static double raiseNegDivByZero(void)
{
    return -1.0/numarray_zero;
}

static double num_log(double x)
{
    if (x == 0.0)
        return raiseNegDivByZero();
    else
        return log(x);
}

static double num_log10(double x)
{
    if (x == 0.0)
        return raiseNegDivByZero();
    else
        return log10(x);
}

static double num_pow(double x, double y)
{
    int z = (int) y;
    if ((x < 0.0) && (y != z))
        return raiseDivByZero();
    else
        return pow(x, y);
}

/* Inverse hyperbolic trig functions from Numeric */
static double num_acosh(double x)
{
    return log(x + sqrt((x-1.0)*(x+1.0)));
}

static double num_asinh(double xx)
{
    double x;
    int sign;
    if (xx < 0.0) {
        sign = -1;
        x = -xx;
    }
    else {
        sign = 1;
        x = xx;
    }
    return sign*log(x + sqrt(x*x+1.0));
}

static double num_atanh(double x)
{
    return 0.5*log((1.0+x)/(1.0-x));
}

/* NUM_CROUND (in numcomplex.h) also calls num_round */
static double num_round(double x)
{
    return (x >= 0) ? floor(x+0.5) : ceil(x-0.5);
}


/* The following routine is used in the event of a detected integer *
 ** divide by zero so that a floating divide by zero is generated.   *
 ** This is done since numarray uses the floating point exception    *
 ** sticky bits to detect errors. The last bit is an attempt to      *
 ** prevent optimization of the divide by zero away, the input value *
 ** should always be 0                                               *
 */

static int int_dividebyzero_error(long NPY_UNUSED(value), long NPY_UNUSED(unused)) {
    double dummy;
    dummy = 1./numarray_zero;
    if (dummy) /* to prevent optimizer from eliminating expression */
        return 0;
    else
        return 1;
}

/* Likewise for Integer overflows */
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__CYGWIN__) || defined(__MINGW32__) || (defined(__FreeBSD__) && (__FreeBSD_version >= 502114))
static int int_overflow_error(Float64 value) { /* For x86_64 */
    feraiseexcept(FE_OVERFLOW);
    return (int) value;
}
#else
static int int_overflow_error(Float64 value) {
    double dummy;
    dummy = pow(1.e10, fabs(value/2));
    if (dummy) /* to prevent optimizer from eliminating expression */
        return (int) value;
    else
        return 1;
}
#endif

static int umult64_overflow(UInt64 a, UInt64 b)
{
    UInt64 ah, al, bh, bl, w, x, y, z;

    ah = (a >> 32);
    al = (a & 0xFFFFFFFFL);
    bh = (b >> 32);
    bl = (b & 0xFFFFFFFFL);

    /* 128-bit product:  z*2**64 + (x+y)*2**32 + w  */
    w = al*bl;
    x = bh*al;
    y = ah*bl;
    z = ah*bh;

    /* *c = ((x + y)<<32) + w; */
    return z || (x>>32) || (y>>32) ||
        (((x & 0xFFFFFFFFL) + (y & 0xFFFFFFFFL) + (w >> 32)) >> 32);
}

static int smult64_overflow(Int64 a0, Int64 b0)
{
    UInt64 a, b;
    UInt64 ah, al, bh, bl, w, x, y, z;

    /* Convert to non-negative quantities */
    if (a0 < 0) { a = -a0; } else { a = a0; }
    if (b0 < 0) { b = -b0; } else { b = b0; }

    ah = (a >> 32);
    al = (a & 0xFFFFFFFFL);
    bh = (b >> 32);
    bl = (b & 0xFFFFFFFFL);

    w = al*bl;
    x = bh*al;
    y = ah*bl;
    z = ah*bh;

    /*
       UInt64 c = ((x + y)<<32) + w;
       if ((a0 < 0) ^ (b0 < 0))
     *c = -c;
     else
     *c = c
     */

    return z || (x>>31) || (y>>31) ||
        (((x & 0xFFFFFFFFL) + (y & 0xFFFFFFFFL) + (w >> 32)) >> 31);
}


static void
NA_Done(void)
{
    return;
}

static PyArrayObject *
NA_NewAll(int ndim, maybelong *shape, NumarrayType type,
        void *buffer, maybelong byteoffset, maybelong bytestride,
        int byteorder, int aligned, int writeable)
{
    PyArrayObject *result = NA_NewAllFromBuffer(
            ndim, shape, type, Py_None, byteoffset, bytestride,
            byteorder, aligned, writeable);

    if (result) {
        if (!NA_NumArrayCheck((PyObject *) result)) {
            PyErr_Format( PyExc_TypeError,
                    "NA_NewAll: non-NumArray result");
            result = NULL;
        } else {
            if (buffer) {
                memcpy(PyArray_DATA(result), buffer, NA_NBYTES(result));
            } else {
                memset(PyArray_DATA(result), 0, NA_NBYTES(result));
            }
        }
    }
    return  result;
}

static PyArrayObject *
NA_NewAllStrides(int ndim, maybelong *shape, maybelong *strides,
        NumarrayType type, void *buffer, maybelong byteoffset,
        int byteorder, int aligned, int writeable)
{
    int i;
    PyArrayObject *result = NA_NewAll(ndim, shape, type, buffer,
            byteoffset, 0,
            byteorder, aligned, writeable);
    for(i=0; i<ndim; i++)
        PyArray_STRIDES(result)[i] = strides[i];
    return result;
}


static PyArrayObject *
NA_New(void *buffer, NumarrayType type, int ndim, ...)
{
    int i;
    maybelong shape[MAXDIM];
    va_list ap;
    va_start(ap, ndim);
    for(i=0; i<ndim; i++)
        shape[i] = va_arg(ap, int);
    va_end(ap);
    return NA_NewAll(ndim, shape, type, buffer, 0, 0,
            NA_ByteOrder(), 1, 1);
}

static PyArrayObject *
NA_Empty(int ndim, maybelong *shape, NumarrayType type)
{
    return NA_NewAll(ndim, shape, type, NULL, 0, 0,
            NA_ByteOrder(), 1, 1);
}


/* Create a new numarray which is initially a C_array, or which
   references a C_array: aligned, !byteswapped, contiguous, ...
   Call with buffer==NULL to allocate storage.
   */
static PyArrayObject *
NA_vNewArray(void *buffer, NumarrayType type, int ndim, maybelong *shape)
{
    return (PyArrayObject *) NA_NewAll(ndim, shape, type, buffer, 0, 0,
            NA_ByteOrder(), 1, 1);
}

static PyArrayObject *
NA_NewArray(void *buffer, NumarrayType type, int ndim, ...)
{
    int i;
    maybelong shape[MAXDIM];
    va_list ap;
    va_start(ap, ndim);
    for(i=0; i<ndim; i++)
        shape[i] = va_arg(ap, int);  /* literals will still be ints */
    va_end(ap);
    return NA_vNewArray(buffer, type, ndim, shape);
}


static PyObject*
NA_ReturnOutput(PyObject *out, PyArrayObject *shadow)
{
    if ((out == Py_None) || (out == NULL)) {
        /* default behavior: return shadow array as the result */
        return (PyObject *) shadow;
    } else {
        PyObject *rval;
        /* specified output behavior: return None */
        /* del(shadow) --> out.copyFrom(shadow) */
        Py_DECREF(shadow);
        Py_INCREF(Py_None);
        rval = Py_None;
        return rval;
    }
}

static long NA_getBufferPtrAndSize(PyObject *buffobj, int readonly, void **ptr)
{
    long rval;
    if (readonly)
        rval = getReadBufferDataPtr(buffobj, ptr);
    else
        rval = getWriteBufferDataPtr(buffobj, ptr);
    return rval;
}


static int NA_checkIo(char *name,
        int wantIn, int wantOut, int gotIn, int gotOut)
{
    if (wantIn != gotIn) {
        PyErr_Format(_Error,
                "%s: wrong # of input buffers. Expected %d.  Got %d.",
                name, wantIn, gotIn);
        return -1;
    }
    if (wantOut != gotOut) {
        PyErr_Format(_Error,
                "%s: wrong # of output buffers. Expected %d.  Got %d.",
                name, wantOut, gotOut);
        return -1;
    }
    return 0;
}

static int NA_checkOneCBuffer(char *name, long niter,
        void *buffer, long bsize, size_t typesize)
{
    Int64 lniter = niter, ltypesize = typesize;

    if (lniter*ltypesize > bsize) {
        PyErr_Format(_Error,
                "%s: access out of buffer. niter=%d typesize=%d bsize=%d",
                name, (int) niter, (int) typesize, (int) bsize);
        return -1;
    }
    if ((typesize <= sizeof(Float64)) && (((long) buffer) % typesize)) {
        PyErr_Format(_Error,
                "%s: buffer not aligned on %d byte boundary.",
                name, (int) typesize);
        return -1;
    }
    return 0;
}


static int NA_checkNCBuffers(char *name, int N, long niter,
        void **buffers, long *bsizes,
        Int8 *typesizes, Int8 *iters)
{
    int i;
    for (i=0; i<N; i++)
        if (NA_checkOneCBuffer(name, iters[i] ? iters[i] : niter,
                    buffers[i], bsizes[i], typesizes[i]))
            return -1;
    return 0;
}

static int NA_checkOneStriding(char *name, long dim, maybelong *shape,
        long offset, maybelong *stride, long buffersize, long itemsize,
        int align)
{
    int i;
    long omin=offset, omax=offset;
    long alignsize = ((size_t)itemsize <= sizeof(Float64) ? (size_t)itemsize : sizeof(Float64));

    if (align && (offset % alignsize)) {
        PyErr_Format(_Error,
                "%s: buffer not aligned on %d byte boundary.",
                name, (int) alignsize);
        return -1;
    }
    for(i=0; i<dim; i++) {
        long strideN = stride[i] * (shape[i]-1);
        long tmax = omax + strideN;
        long tmin = omin + strideN;
        if (shape[i]-1 >= 0) {  /* Skip dimension == 0. */
            omax = MAX(omax, tmax);
            omin = MIN(omin, tmin);
            if (align && (ABS(stride[i]) % alignsize)) {
                PyErr_Format(_Error,
                        "%s: stride %d not aligned on %d byte boundary.",
                        name, (int) stride[i], (int) alignsize);
                return -1;
            }
            if (omax + itemsize > buffersize) {
                PyErr_Format(_Error,
                        "%s: access beyond buffer. offset=%d buffersize=%d",
                        name, (int) (omax+itemsize-1), (int) buffersize);
                return -1;
            }
            if (omin < 0) {
                PyErr_Format(_Error,
                        "%s: access before buffer. offset=%d buffersize=%d",
                        name, (int) omin, (int) buffersize);
                return -1;
            }
        }
    }
    return 0;
}

/* Function to call standard C Ufuncs
 **
 ** The C Ufuncs expect contiguous 1-d data numarray, input and output numarray
 ** iterate with standard increments of one data element over all numarray.
 ** (There are some exceptions like arrayrangexxx which use one or more of
 ** the data numarray as parameter or other sources of information and do not
 ** iterate over every buffer).
 **
 ** Arguments:
 **
 **   Number of iterations (simple integer value).
 **   Number of input numarray.
 **   Number of output numarray.
 **   Tuple of tuples, one tuple per input/output array. Each of these
 **     tuples consists of a buffer object and a byte offset to start.
 **
 ** Returns None
 */


static PyObject *
NA_callCUFuncCore(PyObject *self,
        long niter, long ninargs, long noutargs,
        PyObject **BufferObj, long *offset)
{
    CfuncObject *me = (CfuncObject *) self;
    char *buffers[MAXARGS];
    long bsizes[MAXARGS];
    long i, pnargs = ninargs + noutargs;
    UFUNC ufuncptr;

    if (pnargs > MAXARGS)
        return PyErr_Format(PyExc_RuntimeError, "NA_callCUFuncCore: too many parameters");

    if (!PyObject_IsInstance(self, (PyObject *) &CfuncType)
            || me->descr.type != CFUNC_UFUNC)
        return PyErr_Format(PyExc_TypeError,
                "NA_callCUFuncCore: problem with cfunc.");

    for (i=0; i<pnargs; i++) {
        int readonly = (i < ninargs);
        if (offset[i] < 0)
            return PyErr_Format(_Error,
                    "%s: invalid negative offset:%d for buffer[%d]",
                    me->descr.name, (int) offset[i], (int) i);
        if ((bsizes[i] = NA_getBufferPtrAndSize(BufferObj[i], readonly,
                        (void *) &buffers[i])) < 0)
            return PyErr_Format(_Error,
                    "%s: Problem with %s buffer[%d].",
                    me->descr.name,
                    readonly ? "read" : "write", (int) i);
        buffers[i] += offset[i];
        bsizes[i]  -= offset[i]; /* "shorten" buffer size by offset. */
    }

    ufuncptr = (UFUNC) me->descr.fptr;

    /* If it's not a self-checking ufunc, check arg count match,
       buffer size, and alignment for all buffers */
    if (!me->descr.chkself &&
            (NA_checkIo(me->descr.name,
                        me->descr.wantIn, me->descr.wantOut, ninargs, noutargs) ||
             NA_checkNCBuffers(me->descr.name, pnargs,
                 niter, (void **) buffers, bsizes,
                 me->descr.sizes, me->descr.iters)))
        return NULL;

    /* Since the parameters are valid, call the C Ufunc */
    if (!(*ufuncptr)(niter, ninargs, noutargs, (void **)buffers, bsizes)) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        return NULL;
    }
}

static PyObject *
callCUFunc(PyObject *self, PyObject *args) {
    PyObject *DataArgs, *ArgTuple;
    long pnargs, ninargs, noutargs, niter, i;
    CfuncObject *me = (CfuncObject *) self;
    PyObject *BufferObj[MAXARGS];
    long     offset[MAXARGS];

    if (!PyArg_ParseTuple(args, "lllO",
                &niter, &ninargs, &noutargs, &DataArgs))
        return PyErr_Format(_Error,
                "%s: Problem with argument list", me->descr.name);

    /* check consistency of stated inputs/outputs and supplied buffers */
    pnargs = PyObject_Length(DataArgs);
    if ((pnargs != (ninargs+noutargs)) || (pnargs > MAXARGS))
        return PyErr_Format(_Error,
                "%s: wrong buffer count for function", me->descr.name);

    /* Unpack buffers and offsets, get data pointers */
    for (i=0; i<pnargs; i++) {
        ArgTuple = PySequence_GetItem(DataArgs, i);
        Py_DECREF(ArgTuple);
        if (!PyArg_ParseTuple(ArgTuple, "Ol", &BufferObj[i], &offset[i]))
            return PyErr_Format(_Error,
                    "%s: Problem with buffer/offset tuple", me->descr.name);
    }
    return NA_callCUFuncCore(self, niter, ninargs, noutargs, BufferObj, offset);
}

static PyObject *
callStrideConvCFunc(PyObject *self, PyObject *args) {
    PyObject *inbuffObj, *outbuffObj, *shapeObj;
    PyObject *inbstridesObj, *outbstridesObj;
    CfuncObject *me = (CfuncObject *) self;
    int  nshape, ninbstrides, noutbstrides;
    maybelong shape[MAXDIM], inbstrides[MAXDIM],
              outbstrides[MAXDIM], *outbstrides1 = outbstrides;
    long inboffset, outboffset, nbytes=0;

    if (!PyArg_ParseTuple(args, "OOlOOlO|l",
                &shapeObj, &inbuffObj,  &inboffset, &inbstridesObj,
                &outbuffObj, &outboffset, &outbstridesObj,
                &nbytes)) {
        return PyErr_Format(_Error,
                "%s: Problem with argument list",
                me->descr.name);
    }

    nshape = NA_maybeLongsFromIntTuple(MAXDIM, shape, shapeObj);
    if (nshape < 0) return NULL;

    ninbstrides = NA_maybeLongsFromIntTuple(MAXDIM, inbstrides, inbstridesObj);
    if (ninbstrides < 0) return NULL;

    noutbstrides=  NA_maybeLongsFromIntTuple(MAXDIM, outbstrides, outbstridesObj);
    if (noutbstrides < 0) return NULL;

    if (nshape && (nshape != ninbstrides)) {
        return PyErr_Format(_Error,
                "%s: Missmatch between input iteration and strides tuples",
                me->descr.name);
    }

    if (nshape && (nshape != noutbstrides)) {
        if (noutbstrides < 1 ||
                outbstrides[ noutbstrides - 1 ])/* allow 0 for reductions. */
            return PyErr_Format(_Error,
                    "%s: Missmatch between output "
                    "iteration and strides tuples",
                    me->descr.name);
    }

    return NA_callStrideConvCFuncCore(
            self, nshape, shape,
            inbuffObj,  inboffset,  ninbstrides, inbstrides,
            outbuffObj, outboffset, noutbstrides, outbstrides1, nbytes);
}

static int
_NA_callStridingHelper(PyObject *aux, long dim,
        long nnumarray, PyArrayObject *numarray[], char *data[],
        CFUNC_STRIDED_FUNC f)
{
    int i, j, status=0;
    dim -= 1;
    for(i=0; i<PyArray_DIMS(numarray[0])[dim]; i++) {
        for (j=0; j<nnumarray; j++)
            data[j] += PyArray_STRIDES(numarray[j])[dim]*i;
        if (dim == 0)
            status |= f(aux, nnumarray, numarray, data);
        else
            status |= _NA_callStridingHelper(
                    aux, dim, nnumarray, numarray, data, f);
        for (j=0; j<nnumarray; j++)
            data[j] -= PyArray_STRIDES(numarray[j])[dim]*i;
    }
    return status;
}


static PyObject *
callStridingCFunc(PyObject *self, PyObject *args) {
    CfuncObject *me = (CfuncObject *) self;
    PyObject *aux;
    PyArrayObject *numarray[MAXARRAYS];
    char *data[MAXARRAYS];
    CFUNC_STRIDED_FUNC f;
    int i;

    int nnumarray = PySequence_Length(args)-1;
    if ((nnumarray < 1) || (nnumarray > MAXARRAYS))
        return PyErr_Format(_Error, "%s, too many or too few numarray.",
                me->descr.name);

    aux = PySequence_GetItem(args, 0);
    if (!aux)
        return NULL;

    for(i=0; i<nnumarray; i++) {
        PyObject *otemp = PySequence_GetItem(args, i+1);
        if (!otemp)
            return PyErr_Format(_Error, "%s couldn't get array[%d]",
                    me->descr.name, i);
        if (!NA_NDArrayCheck(otemp))
            return PyErr_Format(PyExc_TypeError,
                    "%s arg[%d] is not an array.",
                    me->descr.name, i);
        numarray[i] = (PyArrayObject *) otemp;
        data[i] = PyArray_DATA(numarray[i]);
        Py_DECREF(otemp);
        if (!NA_updateDataPtr(numarray[i]))
            return NULL;
    }

    /* Cast function pointer and perform stride operation */
    f = (CFUNC_STRIDED_FUNC) me->descr.fptr;

    if (_NA_callStridingHelper(aux, PyArray_NDIM(numarray[0]),
                nnumarray, numarray, data, f)) {
        return NULL;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

/* Convert a standard C numeric value to a Python numeric value.
 **
 ** Handles both nonaligned and/or byteswapped C data.
 **
 ** Input arguments are:
 **
 **   Buffer object that contains the C numeric value.
 **   Offset (in bytes) into the buffer that the data is located at.
 **   The size of the C numeric data item in bytes.
 **   Flag indicating if the C data is byteswapped from the processor's
 **     natural representation.
 **
 **   Returns a Python numeric value.
 */

static PyObject *
NumTypeAsPyValue(PyObject *self, PyObject *args) {
    PyObject *bufferObj;
    long offset, itemsize, byteswap, i, buffersize;
    Py_complex temp;  /* to hold copies of largest possible type */
    void *buffer;
    char *tempptr;
    CFUNCasPyValue funcptr;
    CfuncObject *me = (CfuncObject *) self;

    if (!PyArg_ParseTuple(args, "Olll",
                &bufferObj, &offset, &itemsize, &byteswap))
        return PyErr_Format(_Error,
                "NumTypeAsPyValue: Problem with argument list");

    if ((buffersize = NA_getBufferPtrAndSize(bufferObj, 1, &buffer)) < 0)
        return PyErr_Format(_Error,
                "NumTypeAsPyValue: Problem with array buffer");

    if (offset < 0)
        return PyErr_Format(_Error,
                "NumTypeAsPyValue: invalid negative offset: %d", (int) offset);

    /* Guarantee valid buffer pointer */
    if (offset+itemsize > buffersize)
        return PyErr_Format(_Error,
                "NumTypeAsPyValue: buffer too small for offset and itemsize.");

    /* Do byteswapping.  Guarantee double alignment by using temp. */
    tempptr = (char *) &temp;
    if (!byteswap) {
        for (i=0; i<itemsize; i++)
            *(tempptr++) = *(((char *) buffer)+offset+i);
    } else {
        tempptr += itemsize-1;
        for (i=0; i<itemsize; i++)
            *(tempptr--) = *(((char *) buffer)+offset+i);
    }

    funcptr = (CFUNCasPyValue) me->descr.fptr;

    /* Call function to build PyObject.  Bad parameters to this function
       may render call meaningless, but "temp" guarantees that its safe.  */
    return (*funcptr)((void *)(&temp));
}

/* Convert a Python numeric value to a standard C numeric value.
 **
 ** Handles both nonaligned and/or byteswapped C data.
 **
 ** Input arguments are:
 **
 **   The Python numeric value to be converted.
 **   Buffer object to contain the C numeric value.
 **   Offset (in bytes) into the buffer that the data is to be copied to.
 **   The size of the C numeric data item in bytes.
 **   Flag indicating if the C data is byteswapped from the processor's
 **     natural representation.
 **
 **   Returns None
 */

static PyObject *
NumTypeFromPyValue(PyObject *self, PyObject *args) {
    PyObject *bufferObj, *valueObj;
    long offset, itemsize, byteswap, i, buffersize;
    Py_complex temp;  /* to hold copies of largest possible type */
    void *buffer;
    char *tempptr;
    CFUNCfromPyValue funcptr;
    CfuncObject *me = (CfuncObject *) self;

    if (!PyArg_ParseTuple(args, "OOlll",
                &valueObj, &bufferObj, &offset, &itemsize, &byteswap))
        return PyErr_Format(_Error,
                "%s: Problem with argument list", me->descr.name);

    if ((buffersize = NA_getBufferPtrAndSize(bufferObj, 0, &buffer)) < 0)
        return PyErr_Format(_Error,
                "%s: Problem with array buffer (read only?)", me->descr.name);

    funcptr = (CFUNCfromPyValue) me->descr.fptr;

    /* Convert python object into "temp". Always safe. */
    if (!((*funcptr)(valueObj, (void *)( &temp))))
        return PyErr_Format(_Error,
                "%s: Problem converting value", me->descr.name);

    /* Check buffer offset. */
    if (offset < 0)
        return PyErr_Format(_Error,
                "%s: invalid negative offset: %d", me->descr.name, (int) offset);

    if (offset+itemsize > buffersize)
        return PyErr_Format(_Error,
                "%s: buffer too small(%d) for offset(%d) and itemsize(%d)",
                me->descr.name, (int) buffersize, (int) offset, (int) itemsize);

    /* Copy "temp" to array buffer. */
    tempptr = (char *) &temp;
    if (!byteswap) {
        for (i=0; i<itemsize; i++)
            *(((char *) buffer)+offset+i) = *(tempptr++);
    } else {
        tempptr += itemsize-1;
        for (i=0; i<itemsize; i++)
            *(((char *) buffer)+offset+i) = *(tempptr--);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

/* Handle "calling" the cfunc object at the python level.
   Dispatch the call to the appropriate python-c wrapper based
   on the cfunc type.  Do this dispatch to avoid having to
   check that python code didn't somehow create a mismatch between
   cfunc and wrapper.
   */
static PyObject *
cfunc_call(PyObject *self, PyObject *argsTuple, PyObject *NPY_UNUSED(argsDict))
{
    CfuncObject *me = (CfuncObject *) self;
    switch(me->descr.type) {
    case CFUNC_UFUNC:
        return callCUFunc(self, argsTuple);
        break;
    case CFUNC_STRIDING:
        return callStrideConvCFunc(self, argsTuple);
        break;
    case CFUNC_NSTRIDING:
        return callStridingCFunc(self, argsTuple);
    case CFUNC_FROM_PY_VALUE:
        return NumTypeFromPyValue(self, argsTuple);
        break;
    case CFUNC_AS_PY_VALUE:
        return NumTypeAsPyValue(self, argsTuple);
        break;
    default:
        return PyErr_Format( _Error,
                "cfunc_call: Can't dispatch cfunc '%s' with type: %d.",
                me->descr.name, me->descr.type);
    }
}

static PyTypeObject CfuncType;

static void
cfunc_dealloc(PyObject* self)
{
    PyObject_Del(self);
}

static PyObject *
cfunc_repr(PyObject *self)
{
    char buf[256];
    CfuncObject *me = (CfuncObject *) self;
    sprintf(buf, "<cfunc '%s' at %08lx check-self:%d align:%d  io:(%d, %d)>",
            me->descr.name, (unsigned long ) me->descr.fptr,
            me->descr.chkself, me->descr.align,
            me->descr.wantIn, me->descr.wantOut);
    return PyUString_FromString(buf);
}

static PyTypeObject CfuncType = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(0,0)
#else
    PyObject_HEAD_INIT(0)
    0,                                          /* ob_size */
#endif
    "Cfunc",
    sizeof(CfuncObject),
    0,
    cfunc_dealloc,                              /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    cfunc_repr,                                 /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    cfunc_call,                                 /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
    };

/* CfuncObjects are created at the c-level only.  They ensure that each
   cfunc is called via the correct python-c-wrapper as defined by its
   CfuncDescriptor.  The wrapper, in turn, does conversions and buffer size
   and alignment checking.  Allowing these to be created at the python level
   would enable them to be created *wrong* at the python level, and thereby
   enable python code to *crash* python.
   */
static PyObject*
NA_new_cfunc(CfuncDescriptor *cfd)
{
    CfuncObject* cfunc;

    /* Should be done once at init.
       Do now since there is no init. */
    ((PyObject*)&CfuncType)->ob_type = &PyType_Type;

    cfunc = PyObject_New(CfuncObject, &CfuncType);

    if (!cfunc) {
        return PyErr_Format(_Error,
                "NA_new_cfunc: failed creating '%s'",
                cfd->name);
    }

    cfunc->descr = *cfd;

    return (PyObject*)cfunc;
}

static int NA_add_cfunc(PyObject *dict, char *keystr, CfuncDescriptor *descr)
{
    PyObject *c = (PyObject *) NA_new_cfunc(descr);
    if (!c) return -1;
    return PyDict_SetItemString(dict, keystr, c);
}

static PyArrayObject*
NA_InputArray(PyObject *a, NumarrayType t, int requires)
{
    PyArray_Descr *descr;
    if (t == tAny) descr = NULL;
    else descr = PyArray_DescrFromType(t);
    return (PyArrayObject *)                                    \
        PyArray_CheckFromAny(a, descr, 0, 0, requires, NULL);
}

/* satisfies ensures that 'a' meets a set of requirements and matches
   the specified type.
   */
static int
satisfies(PyArrayObject *a, int requirements, NumarrayType t)
{
    int type_ok = (PyArray_DESCR(a)->type_num == t) || (t == tAny);

    if (PyArray_ISCARRAY(a))
        return type_ok;
    if (PyArray_ISBYTESWAPPED(a) && (requirements & NUM_NOTSWAPPED))
        return 0;
    if (!PyArray_ISALIGNED(a) && (requirements & NUM_ALIGNED))
        return 0;
    if (!PyArray_ISCONTIGUOUS(a) && (requirements & NUM_CONTIGUOUS))
        return 0;
    if (!PyArray_ISWRITABLE(a) && (requirements & NUM_WRITABLE))
        return 0;
    if (requirements & NUM_COPY)
        return 0;
    return type_ok;
}


static PyArrayObject *
NA_OutputArray(PyObject *a, NumarrayType t, int requires)
{
    PyArray_Descr *dtype;
    PyArrayObject *ret;

    if (!PyArray_Check(a)) {
        PyErr_Format(PyExc_TypeError,
                "NA_OutputArray: only arrays work for output.");
        return NULL;
    }
    if (PyArray_FailUnlessWriteable((PyArrayObject *)a, "output array") < 0) {
        return NULL;
    }

    if (satisfies((PyArrayObject *)a, requires, t)) {
        Py_INCREF(a);
        return (PyArrayObject *)a;
    }
    if (t == tAny) {
        dtype = PyArray_DESCR((PyArrayObject *)a);
        Py_INCREF(dtype);
    }
    else {
        dtype = PyArray_DescrFromType(t);
    }
    ret = (PyArrayObject *)PyArray_Empty(PyArray_NDIM((PyArrayObject *)a),
                                        PyArray_DIMS((PyArrayObject *)a),
                                        dtype, 0);
    Py_INCREF(a);
    if (PyArray_SetUpdateIfCopyBase(ret, a) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return ret;
}


/* NA_IoArray is a combination of NA_InputArray and NA_OutputArray.

   Unlike NA_OutputArray, if a temporary is required it is initialized to a copy
   of the input array.

   Unlike NA_InputArray, deallocating any resulting temporary array results in a
   copy from the temporary back to the original.
   */
static PyArrayObject *
NA_IoArray(PyObject *a, NumarrayType t, int requires)
{
    PyArrayObject *shadow = NA_InputArray(a, t,
                                requires | NPY_ARRAY_UPDATEIFCOPY );

    if (!shadow) return NULL;

    /* Guard against non-writable, but otherwise satisfying requires.
       In this case,  shadow == a.
       */
    if (!PyArray_FailUnlessWriteable(shadow, "input/output array")) {
        PyArray_XDECREF_ERR(shadow);
        return NULL;
    }

    return shadow;
}

/* NA_OptionalOutputArray works like NA_OutputArray, but handles the case
   where the output array 'optional' is omitted entirely at the python level,
   resulting in 'optional'==Py_None.  When 'optional' is Py_None, the return
   value is cloned (but with NumarrayType 't') from 'master', typically an input
   array with the same shape as the output array.
   */
static PyArrayObject *
NA_OptionalOutputArray(PyObject *optional, NumarrayType t, int requires,
        PyArrayObject *master)
{
    if ((optional == Py_None) || (optional == NULL)) {
        PyObject *rval;
        PyArray_Descr *descr;
        if (t == tAny) descr=NULL;
        else descr = PyArray_DescrFromType(t);
        rval = PyArray_FromArray(
                master, descr, NUM_C_ARRAY | NUM_COPY | NUM_WRITABLE);
        return (PyArrayObject *)rval;
    } else {
        return NA_OutputArray(optional, t, requires);
    }
}

Complex64 NA_get_Complex64(PyArrayObject *a, long offset)
{
    Complex32 v0;
    Complex64 v;

    switch(PyArray_DESCR(a)->type_num) {
    case tComplex32:
        v0 = NA_GETP(a, Complex32, (NA_PTR(a)+offset));
        v.r = v0.r;
        v.i = v0.i;
        break;
    case tComplex64:
        v = NA_GETP(a, Complex64, (NA_PTR(a)+offset));
        break;
    default:
        v.r = NA_get_Float64(a, offset);
        v.i = 0;
        break;
    }
    return v;
}

void NA_set_Complex64(PyArrayObject *a, long offset, Complex64 v)
{
    Complex32 v0;

    switch(PyArray_DESCR(a)->type_num) {
    case tComplex32:
        v0.r = v.r;
        v0.i = v.i;
        NA_SETP(a, Complex32, (NA_PTR(a)+offset), v0);
        break;
    case tComplex64:
        NA_SETP(a, Complex64, (NA_PTR(a)+offset), v);
        break;
    default:
        NA_set_Float64(a, offset, v.r);
        break;
    }
}

Int64 NA_get_Int64(PyArrayObject *a, long offset)
{
    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        return NA_GETP(a, Bool, (NA_PTR(a)+offset)) != 0;
    case tInt8:
        return NA_GETP(a, Int8, (NA_PTR(a)+offset));
    case tUInt8:
        return NA_GETP(a, UInt8, (NA_PTR(a)+offset));
    case tInt16:
        return NA_GETP(a, Int16, (NA_PTR(a)+offset));
    case tUInt16:
        return NA_GETP(a, UInt16, (NA_PTR(a)+offset));
    case tInt32:
        return NA_GETP(a, Int32, (NA_PTR(a)+offset));
    case tUInt32:
        return NA_GETP(a, UInt32, (NA_PTR(a)+offset));
    case tInt64:
        return NA_GETP(a, Int64, (NA_PTR(a)+offset));
    case tUInt64:
        return NA_GETP(a, UInt64, (NA_PTR(a)+offset));
    case tFloat32:
        return NA_GETP(a, Float32, (NA_PTR(a)+offset));
    case tFloat64:
        return NA_GETP(a, Float64, (NA_PTR(a)+offset));
    case tComplex32:
        return NA_GETP(a, Float32, (NA_PTR(a)+offset));
    case tComplex64:
        return NA_GETP(a, Float64, (NA_PTR(a)+offset));
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_get_Int64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
    }
    return 0; /* suppress warning */
}

void NA_set_Int64(PyArrayObject *a, long offset, Int64 v)
{
    Bool b;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        b = (v != 0);
        NA_SETP(a, Bool, (NA_PTR(a)+offset), b);
        break;
    case tInt8:    NA_SETP(a, Int8, (NA_PTR(a)+offset), v);
                   break;
    case tUInt8:   NA_SETP(a, UInt8, (NA_PTR(a)+offset), v);
                   break;
    case tInt16:   NA_SETP(a, Int16, (NA_PTR(a)+offset), v);
                   break;
    case tUInt16:  NA_SETP(a, UInt16, (NA_PTR(a)+offset), v);
                   break;
    case tInt32:   NA_SETP(a, Int32, (NA_PTR(a)+offset), v);
                   break;
    case tUInt32:   NA_SETP(a, UInt32, (NA_PTR(a)+offset), v);
                    break;
    case tInt64:   NA_SETP(a, Int64, (NA_PTR(a)+offset), v);
                   break;
    case tUInt64:   NA_SETP(a, UInt64, (NA_PTR(a)+offset), v);
                    break;
    case tFloat32:
                    NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
                    break;
    case tFloat64:
                    NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
                    break;
    case tComplex32:
                    NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
                    NA_SETP(a, Float32, (NA_PTR(a)+offset+sizeof(Float32)), 0);
                    break;
    case tComplex64:
                    NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
                    NA_SETP(a, Float64, (NA_PTR(a)+offset+sizeof(Float64)), 0);
                    break;
    default:
                    PyErr_Format( PyExc_TypeError,
                            "Unknown type %d in NA_set_Int64",
                            PyArray_DESCR(a)->type_num);
                    PyErr_Print();
    }
}

/*  NA_get_offset computes the offset specified by the set of indices.
    If N > 0, the indices are taken from the outer dimensions of the array.
    If N < 0, the indices are taken from the inner dimensions of the array.
    If N == 0, the offset is 0.
    */
long NA_get_offset(PyArrayObject *a, int N, ...)
{
    int i;
    long offset = 0;
    va_list ap;
    va_start(ap, N);
    if (N > 0) { /* compute offset of "outer" indices. */
        for(i=0; i<N; i++)
            offset += va_arg(ap, long) * PyArray_STRIDES(a)[i];
    } else {   /* compute offset of "inner" indices. */
        N = -N;
        for(i=0; i<N; i++)
            offset += va_arg(ap, long) * PyArray_STRIDES(a)[PyArray_NDIM(a)-N+i];
    }
    va_end(ap);
    return offset;
}

Float64 NA_get_Float64(PyArrayObject *a, long offset)
{
    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        return NA_GETP(a, Bool, (NA_PTR(a)+offset)) != 0;
    case tInt8:
        return NA_GETP(a, Int8, (NA_PTR(a)+offset));
    case tUInt8:
        return NA_GETP(a, UInt8, (NA_PTR(a)+offset));
    case tInt16:
        return NA_GETP(a, Int16, (NA_PTR(a)+offset));
    case tUInt16:
        return NA_GETP(a, UInt16, (NA_PTR(a)+offset));
    case tInt32:
        return NA_GETP(a, Int32, (NA_PTR(a)+offset));
    case tUInt32:
        return NA_GETP(a, UInt32, (NA_PTR(a)+offset));
    case tInt64:
        return NA_GETP(a, Int64, (NA_PTR(a)+offset));
#if HAS_UINT64
    case tUInt64:
        return NA_GETP(a, UInt64, (NA_PTR(a)+offset));
#endif
    case tFloat32:
        return NA_GETP(a, Float32, (NA_PTR(a)+offset));
    case tFloat64:
        return NA_GETP(a, Float64, (NA_PTR(a)+offset));
    case tComplex32:  /* Since real value is first */
        return NA_GETP(a, Float32, (NA_PTR(a)+offset));
    case tComplex64:  /* Since real value is first */
        return NA_GETP(a, Float64, (NA_PTR(a)+offset));
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_get_Float64",
                PyArray_DESCR(a)->type_num);
    }
    return 0; /* suppress warning */
}

void NA_set_Float64(PyArrayObject *a, long offset, Float64 v)
{
    Bool b;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        b = (v != 0);
        NA_SETP(a, Bool, (NA_PTR(a)+offset), b);
        break;
    case tInt8:    NA_SETP(a, Int8, (NA_PTR(a)+offset), v);
                   break;
    case tUInt8:   NA_SETP(a, UInt8, (NA_PTR(a)+offset), v);
                   break;
    case tInt16:   NA_SETP(a, Int16, (NA_PTR(a)+offset), v);
                   break;
    case tUInt16:  NA_SETP(a, UInt16, (NA_PTR(a)+offset), v);
                   break;
    case tInt32:   NA_SETP(a, Int32, (NA_PTR(a)+offset), v);
                   break;
    case tUInt32:   NA_SETP(a, UInt32, (NA_PTR(a)+offset), v);
                    break;
    case tInt64:   NA_SETP(a, Int64, (NA_PTR(a)+offset), v);
                   break;
#if HAS_UINT64
    case tUInt64:   NA_SETP(a, UInt64, (NA_PTR(a)+offset), v);
                    break;
#endif
    case tFloat32:
                    NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
                    break;
    case tFloat64:
                    NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
                    break;
    case tComplex32: {
                         NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
                         NA_SETP(a, Float32, (NA_PTR(a)+offset+sizeof(Float32)), 0);
                         break;
                     }
    case tComplex64: {
                         NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
                         NA_SETP(a, Float64, (NA_PTR(a)+offset+sizeof(Float64)), 0);
                         break;
                     }
    default:
                     PyErr_Format( PyExc_TypeError,
                             "Unknown type %d in NA_set_Float64",
                             PyArray_DESCR(a)->type_num );
                     PyErr_Print();
    }
}


Float64 NA_get1_Float64(PyArrayObject *a, long i)
{
    long offset = i * PyArray_STRIDES(a)[0];
    return NA_get_Float64(a, offset);
}

Float64 NA_get2_Float64(PyArrayObject *a, long i, long j)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    return NA_get_Float64(a, offset);
}

Float64 NA_get3_Float64(PyArrayObject *a, long i, long j, long k)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    return NA_get_Float64(a, offset);
}

void NA_set1_Float64(PyArrayObject *a, long i, Float64 v)
{
    long offset = i * PyArray_STRIDES(a)[0];
    NA_set_Float64(a, offset, v);
}

void NA_set2_Float64(PyArrayObject *a, long i, long j, Float64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    NA_set_Float64(a, offset, v);
}

void NA_set3_Float64(PyArrayObject *a, long i, long j, long k, Float64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    NA_set_Float64(a, offset, v);
}

Complex64 NA_get1_Complex64(PyArrayObject *a, long i)
{
    long offset = i * PyArray_STRIDES(a)[0];
    return NA_get_Complex64(a, offset);
}

Complex64 NA_get2_Complex64(PyArrayObject *a, long i, long j)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    return NA_get_Complex64(a, offset);
}

Complex64 NA_get3_Complex64(PyArrayObject *a, long i, long j, long k)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    return NA_get_Complex64(a, offset);
}

void NA_set1_Complex64(PyArrayObject *a, long i, Complex64 v)
{
    long offset = i * PyArray_STRIDES(a)[0];
    NA_set_Complex64(a, offset, v);
}

void NA_set2_Complex64(PyArrayObject *a, long i, long j, Complex64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    NA_set_Complex64(a, offset, v);
}

void NA_set3_Complex64(PyArrayObject *a, long i, long j, long k, Complex64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    NA_set_Complex64(a, offset, v);
}

Int64 NA_get1_Int64(PyArrayObject *a, long i)
{
    long offset = i * PyArray_STRIDES(a)[0];
    return NA_get_Int64(a, offset);
}

Int64 NA_get2_Int64(PyArrayObject *a, long i, long j)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    return NA_get_Int64(a, offset);
}

Int64 NA_get3_Int64(PyArrayObject *a, long i, long j, long k)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    return NA_get_Int64(a, offset);
}

void NA_set1_Int64(PyArrayObject *a, long i, Int64 v)
{
    long offset = i * PyArray_STRIDES(a)[0];
    NA_set_Int64(a, offset, v);
}

void NA_set2_Int64(PyArrayObject *a, long i, long j, Int64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1];
    NA_set_Int64(a, offset, v);
}

void NA_set3_Int64(PyArrayObject *a, long i, long j, long k, Int64 v)
{
    long offset  = i * PyArray_STRIDES(a)[0]
        + j * PyArray_STRIDES(a)[1]
        + k * PyArray_STRIDES(a)[2];
    NA_set_Int64(a, offset, v);
}

/* SET_CMPLX could be made faster by factoring it into 3 seperate loops.
*/
#define NA_SET_CMPLX(a, type, base, cnt, in)                                  \
{                                                                             \
    int i;                                                                \
    int stride = PyArray_STRIDES(a)[ PyArray_NDIM(a) - 1];                                  \
    NA_SET1D(a, type, base, cnt, in);                                     \
    base = NA_PTR(a) + offset + sizeof(type);                             \
    for(i=0; i<cnt; i++) {                                                \
        NA_SETP(a, Float32, base, 0);                                 \
        base += stride;                                               \
    }                                                                     \
}

static int
NA_get1D_Float64(PyArrayObject *a, long offset, int cnt, Float64*out)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        NA_GET1D(a, Bool, base, cnt, out);
        break;
    case tInt8:
        NA_GET1D(a, Int8, base, cnt, out);
        break;
    case tUInt8:
        NA_GET1D(a, UInt8, base, cnt, out);
        break;
    case tInt16:
        NA_GET1D(a, Int16, base, cnt, out);
        break;
    case tUInt16:
        NA_GET1D(a, UInt16, base, cnt, out);
        break;
    case tInt32:
        NA_GET1D(a, Int32, base, cnt, out);
        break;
    case tUInt32:
        NA_GET1D(a, UInt32, base, cnt, out);
        break;
    case tInt64:
        NA_GET1D(a, Int64, base, cnt, out);
        break;
#if HAS_UINT64
    case tUInt64:
        NA_GET1D(a, UInt64, base, cnt, out);
        break;
#endif
    case tFloat32:
        NA_GET1D(a, Float32, base, cnt, out);
        break;
    case tFloat64:
        NA_GET1D(a, Float64, base, cnt, out);
        break;
    case tComplex32:
        NA_GET1D(a, Float32, base, cnt, out);
        break;
    case tComplex64:
        NA_GET1D(a, Float64, base, cnt, out);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_get1D_Float64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}

static Float64 *
NA_alloc1D_Float64(PyArrayObject *a, long offset, int cnt)
{
    Float64 *result = PyMem_New(Float64, (size_t)cnt);
    if (!result) return NULL;
    if (NA_get1D_Float64(a, offset, cnt, result) < 0) {
        PyMem_Free(result);
        return NULL;
    }
    return result;
}

static int
NA_set1D_Float64(PyArrayObject *a, long offset, int cnt, Float64*in)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        NA_SET1D(a, Bool, base, cnt, in);
        break;
    case tInt8:
        NA_SET1D(a, Int8, base, cnt, in);
        break;
    case tUInt8:
        NA_SET1D(a, UInt8, base, cnt, in);
        break;
    case tInt16:
        NA_SET1D(a, Int16, base, cnt, in);
        break;
    case tUInt16:
        NA_SET1D(a, UInt16, base, cnt, in);
        break;
    case tInt32:
        NA_SET1D(a, Int32, base, cnt, in);
        break;
    case tUInt32:
        NA_SET1D(a, UInt32, base, cnt, in);
        break;
    case tInt64:
        NA_SET1D(a, Int64, base, cnt, in);
        break;
#if HAS_UINT64
    case tUInt64:
        NA_SET1D(a, UInt64, base, cnt, in);
        break;
#endif
    case tFloat32:
        NA_SET1D(a, Float32, base, cnt, in);
        break;
    case tFloat64:
        NA_SET1D(a, Float64, base, cnt, in);
        break;
    case tComplex32:
        NA_SET_CMPLX(a, Float32, base, cnt, in);
        break;
    case tComplex64:
        NA_SET_CMPLX(a, Float64, base, cnt, in);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_set1D_Float64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}

static int
NA_get1D_Int64(PyArrayObject *a, long offset, int cnt, Int64*out)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        NA_GET1D(a, Bool, base, cnt, out);
        break;
    case tInt8:
        NA_GET1D(a, Int8, base, cnt, out);
        break;
    case tUInt8:
        NA_GET1D(a, UInt8, base, cnt, out);
        break;
    case tInt16:
        NA_GET1D(a, Int16, base, cnt, out);
        break;
    case tUInt16:
        NA_GET1D(a, UInt16, base, cnt, out);
        break;
    case tInt32:
        NA_GET1D(a, Int32, base, cnt, out);
        break;
    case tUInt32:
        NA_GET1D(a, UInt32, base, cnt, out);
        break;
    case tInt64:
        NA_GET1D(a, Int64, base, cnt, out);
        break;
    case tUInt64:
        NA_GET1D(a, UInt64, base, cnt, out);
        break;
    case tFloat32:
        NA_GET1D(a, Float32, base, cnt, out);
        break;
    case tFloat64:
        NA_GET1D(a, Float64, base, cnt, out);
        break;
    case tComplex32:
        NA_GET1D(a, Float32, base, cnt, out);
        break;
    case tComplex64:
        NA_GET1D(a, Float64, base, cnt, out);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_get1D_Int64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}

static Int64 *
NA_alloc1D_Int64(PyArrayObject *a, long offset, int cnt)
{
    Int64 *result = PyMem_New(Int64, (size_t)cnt);
    if (!result) return NULL;
    if (NA_get1D_Int64(a, offset, cnt, result) < 0) {
        PyMem_Free(result);
        return NULL;
    }
    return result;
}

static int
NA_set1D_Int64(PyArrayObject *a, long offset, int cnt, Int64*in)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        NA_SET1D(a, Bool, base, cnt, in);
        break;
    case tInt8:
        NA_SET1D(a, Int8, base, cnt, in);
        break;
    case tUInt8:
        NA_SET1D(a, UInt8, base, cnt, in);
        break;
    case tInt16:
        NA_SET1D(a, Int16, base, cnt, in);
        break;
    case tUInt16:
        NA_SET1D(a, UInt16, base, cnt, in);
        break;
    case tInt32:
        NA_SET1D(a, Int32, base, cnt, in);
        break;
    case tUInt32:
        NA_SET1D(a, UInt32, base, cnt, in);
        break;
    case tInt64:
        NA_SET1D(a, Int64, base, cnt, in);
        break;
    case tUInt64:
        NA_SET1D(a, UInt64, base, cnt, in);
        break;
    case tFloat32:
        NA_SET1D(a, Float32, base, cnt, in);
        break;
    case tFloat64:
        NA_SET1D(a, Float64, base, cnt, in);
        break;
    case tComplex32:
        NA_SET_CMPLX(a, Float32, base, cnt, in);
        break;
    case tComplex64:
        NA_SET_CMPLX(a, Float64, base, cnt, in);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_set1D_Int64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}

static int
NA_get1D_Complex64(PyArrayObject *a, long offset, int cnt, Complex64*out)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tComplex64:
        NA_GET1D(a, Complex64, base, cnt, out);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unsupported type %d in NA_get1D_Complex64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}

static int
NA_set1D_Complex64(PyArrayObject *a, long offset, int cnt, Complex64*in)
{
    char *base = NA_PTR(a) + offset;

    switch(PyArray_DESCR(a)->type_num) {
    case tComplex64:
        NA_SET1D(a, Complex64, base, cnt, in);
        break;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unsupported type %d in NA_set1D_Complex64",
                PyArray_DESCR(a)->type_num);
        PyErr_Print();
        return -1;
    }
    return 0;
}


/* NA_ShapeEqual returns 1 if 'a' and 'b' have the same shape, 0 otherwise.
*/
static int
NA_ShapeEqual(PyArrayObject *a, PyArrayObject *b)
{
    int i;

    if (!NA_NDArrayCheck((PyObject *) a) ||
            !NA_NDArrayCheck((PyObject*) b)) {
        PyErr_Format(
                PyExc_TypeError,
                "NA_ShapeEqual: non-array as parameter.");
        return -1;
    }
    if (PyArray_NDIM(a) != PyArray_NDIM(b))
        return 0;
    for(i=0; i<PyArray_NDIM(a); i++)
        if (PyArray_DIMS(a)[i] != PyArray_DIMS(b)[i])
            return 0;
    return 1;
}

/* NA_ShapeLessThan returns 1 if a.shape[i] < b.shape[i] for all i, else 0.
   If they have a different number of dimensions, it compares the innermost
   overlapping dimensions of each.
   */
static int
NA_ShapeLessThan(PyArrayObject *a, PyArrayObject *b)
{
    int i;
    int mindim, aoff, boff;
    if (!NA_NDArrayCheck((PyObject *) a) ||
            !NA_NDArrayCheck((PyObject *) b)) {
        PyErr_Format(PyExc_TypeError,
                "NA_ShapeLessThan: non-array as parameter.");
        return -1;
    }
    mindim = MIN(PyArray_NDIM(a), PyArray_NDIM(b));
    aoff = PyArray_NDIM(a) - mindim;
    boff = PyArray_NDIM(b) - mindim;
    for(i=0; i<mindim; i++)
        if (PyArray_DIMS(a)[i+aoff] >=  PyArray_DIMS(b)[i+boff])
            return 0;
    return 1;
}

static int
NA_ByteOrder(void)
{
    unsigned long byteorder_test;
    byteorder_test = 1;
    if (*((char *) &byteorder_test))
        return NUM_LITTLE_ENDIAN;
    else
        return NUM_BIG_ENDIAN;
}

static Bool
NA_IeeeSpecial32( Float32 *f, Int32 *mask)
{
    return NA_IeeeMask32(*f, *mask);
}

static Bool
NA_IeeeSpecial64( Float64 *f, Int32 *mask)
{
    return NA_IeeeMask64(*f, *mask);
}

static PyArrayObject *
NA_updateDataPtr(PyArrayObject *me)
{
    return me;
}


#define ELEM(x) (sizeof(x)/sizeof(x[0]))

typedef struct
{
    char *name;
    int typeno;
} NumarrayTypeNameMapping;

static NumarrayTypeNameMapping NumarrayTypeNameMap[] = {
    {"Any", tAny},
    {"Bool", tBool},
    {"Int8", tInt8},
    {"UInt8", tUInt8},
    {"Int16", tInt16},
    {"UInt16", tUInt16},
    {"Int32", tInt32},
    {"UInt32", tUInt32},
    {"Int64", tInt64},
    {"UInt64", tUInt64},
    {"Float32", tFloat32},
    {"Float64", tFloat64},
    {"Complex32", tComplex32},
    {"Complex64", tComplex64},
    {"Object", tObject},
    {"Long", tLong},
};


/* Convert NumarrayType 'typeno' into the string of the type's name. */
static char *
NA_typeNoToName(int typeno)
{
    size_t i;
    PyObject *typeObj;
    int typeno2;

    for(i=0; i<ELEM(NumarrayTypeNameMap); i++)
        if (typeno == NumarrayTypeNameMap[i].typeno)
            return NumarrayTypeNameMap[i].name;

    /* Handle Numeric typecodes */
    typeObj = NA_typeNoToTypeObject(typeno);
    if (!typeObj) return 0;
    typeno2 = NA_typeObjectToTypeNo(typeObj);
    Py_DECREF(typeObj);

    return NA_typeNoToName(typeno2);
}

/* Look up the NumarrayType which corresponds to typename */

static int
NA_nameToTypeNo(char *typename)
{
    size_t i;
    for(i=0; i<ELEM(NumarrayTypeNameMap); i++)
        if (!strcmp(typename, NumarrayTypeNameMap[i].name))
            return NumarrayTypeNameMap[i].typeno;
    return -1;
}

static PyObject *
getTypeObject(NumarrayType type)
{
    return (PyObject *)PyArray_DescrFromType(type);
}


static PyObject *
NA_typeNoToTypeObject(int typeno)
{
    PyObject *o;
    o = getTypeObject(typeno);
    if (o) Py_INCREF(o);
    return o;
}


static PyObject *
NA_intTupleFromMaybeLongs(int len, maybelong *Longs)
{
    return PyArray_IntTupleFromIntp(len, Longs);
}

static long
NA_maybeLongsFromIntTuple(int len, maybelong *arr, PyObject *sequence)
{
    return PyArray_IntpFromSequence(sequence, arr, len);
}


static int
NA_intTupleProduct(PyObject  *shape, long *prod)
{
    int i, nshape, rval = -1;

    if (!PySequence_Check(shape)) {
        PyErr_Format(PyExc_TypeError,
                "NA_intSequenceProduct: object is not a sequence.");
        goto _exit;
    }
    nshape = PySequence_Size(shape);

    for(i=0, *prod=1;  i<nshape; i++) {
        PyObject *obj = PySequence_GetItem(shape, i);
        if (!obj || !(PyInt_Check(obj) || PyLong_Check(obj))) {
            PyErr_Format(PyExc_TypeError,
                    "NA_intTupleProduct: non-integer in shape.");
            Py_XDECREF(obj);
            goto _exit;
        }
        *prod *= PyInt_AsLong(obj);
        Py_DECREF(obj);
        if (PyErr_Occurred())
            goto _exit;
    }
    rval = 0;
_exit:
    return rval;
}

static long
NA_isIntegerSequence(PyObject *sequence)
{
    PyObject *o;
    long i, size, isInt = 1;
    if (!sequence) {
        isInt = -1;
        goto _exit;
    }
    if (!PySequence_Check(sequence)) {
        isInt = 0;
        goto _exit;
    }
    if ((size = PySequence_Length(sequence)) < 0) {
        isInt = -1;
        goto _exit;
    }
    for(i=0; i<size; i++) {
        o = PySequence_GetItem(sequence, i);
        if (!PyInt_Check(o) && !PyLong_Check(o)) {
            isInt = 0;
            Py_XDECREF(o);
            goto _exit;
        }
        Py_XDECREF(o);
    }
_exit:
    return isInt;
}

static int
getShape(PyObject *a, maybelong *shape, int dims)
{
    long slen;
    if (PyBytes_Check(a)) {
        PyErr_Format(PyExc_TypeError,
                "getShape: numerical sequences can't contain strings.");
        return -1;
    }

    if (!PySequence_Check(a) ||
            (NA_NDArrayCheck(a) && (PyArray_NDIM(PyArray(a)) == 0)))
        return dims;
    slen = PySequence_Length(a);
    if (slen < 0) {
        PyErr_Format(_Error,
                "getShape: couldn't get sequence length.");
        return -1;
    }
    if (!slen) {
        *shape = 0;
        return dims+1;
    } else if (dims < MAXDIM) {
        PyObject *item0 = PySequence_GetItem(a, 0);
        if (item0) {
            *shape = PySequence_Length(a);
            dims = getShape(item0, ++shape, dims+1);
            Py_DECREF(item0);
        } else {
            PyErr_Format(_Error,
                    "getShape: couldn't get sequence item.");
            return -1;
        }
    } else {
        PyErr_Format(_Error,
                "getShape: sequence object nested more than MAXDIM deep.");
        return -1;
    }
    return dims;
}



typedef enum {
    NOTHING,
    NUMBER,
    SEQUENCE
} SequenceConstraint;

static int
setArrayFromSequence(PyArrayObject *a, PyObject *s, int dim, long offset)
{
    SequenceConstraint mustbe = NOTHING;
    int i, seqlen=-1, slen = PySequence_Length(s);

    if (dim > PyArray_NDIM(a)) {
        PyErr_Format(PyExc_ValueError,
                "setArrayFromSequence: sequence/array dimensions mismatch.");
        return -1;
    }

    if (slen != PyArray_DIMS(a)[dim]) {
        PyErr_Format(PyExc_ValueError,
                "setArrayFromSequence: sequence/array shape mismatch.");
        return -1;
    }

    for(i=0; i<slen; i++) {
        PyObject *o = PySequence_GetItem(s, i);
        if (!o) {
            PyErr_SetString(_Error,
                    "setArrayFromSequence: Can't get a sequence item");
            return -1;
        } else if ((NA_isPythonScalar(o) ||
                    (NA_NumArrayCheck(o) && PyArray_NDIM(PyArray(o)) == 0)) &&
                ((mustbe == NOTHING) || (mustbe == NUMBER))) {
            if (NA_setFromPythonScalar(a, offset, o) < 0)
                return -2;
            mustbe = NUMBER;
        } else if (PyBytes_Check(o)) {
            PyErr_SetString( PyExc_ValueError,
                    "setArrayFromSequence: strings can't define numeric numarray.");
            return -3;
        } else if (PySequence_Check(o)) {

            if ((mustbe == NOTHING) || (mustbe == SEQUENCE)) {
                if (mustbe == NOTHING) {
                    mustbe = SEQUENCE;
                    seqlen = PySequence_Length(o);
                } else if (PySequence_Length(o) != seqlen) {
                    PyErr_SetString(
                            PyExc_ValueError,
                            "Nested sequences with different lengths.");
                    return -5;
                }
                setArrayFromSequence(a, o, dim+1, offset);
            } else {
                PyErr_SetString(PyExc_ValueError,
                        "Nested sequences with different lengths.");
                return -4;
            }
        } else {
            PyErr_SetString(PyExc_ValueError, "Invalid sequence.");
            return -6;
        }
        Py_DECREF(o);
        offset += PyArray_STRIDES(a)[dim];
    }
    return 0;
}

static PyObject *
NA_setArrayFromSequence(PyArrayObject *a, PyObject *s)
{
    maybelong shape[MAXDIM];

    if (!PySequence_Check(s))
        return PyErr_Format( PyExc_TypeError,
                "NA_setArrayFromSequence: (array, seq) expected.");

    if (getShape(s, shape, 0) < 0)
        return NULL;

    if (!NA_updateDataPtr(a))
        return NULL;

    if (setArrayFromSequence(a, s, 0, 0) < 0)
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

enum {
    BOOL_SCALAR,
    INT_SCALAR,
    LONG_SCALAR,
    FLOAT_SCALAR,
    COMPLEX_SCALAR
};


static int
_NA_maxType(PyObject *seq, int limit)
{
    if (limit > MAXDIM) {
        PyErr_Format( PyExc_ValueError,
                "NA_maxType: sequence nested too deep." );
        return -1;
    }
    if (NA_NumArrayCheck(seq)) {
        switch(PyArray_DESCR(PyArray(seq))->type_num) {
        case tBool:
            return BOOL_SCALAR;
        case tInt8:
        case tUInt8:
        case tInt16:
        case tUInt16:
        case tInt32:
        case tUInt32:
            return INT_SCALAR;
        case tInt64:
        case tUInt64:
            return LONG_SCALAR;
        case tFloat32:
        case tFloat64:
            return FLOAT_SCALAR;
        case tComplex32:
        case tComplex64:
            return COMPLEX_SCALAR;
        default:
            PyErr_Format(PyExc_TypeError,
                    "Expecting a python numeric type, got something else.");
            return -1;
        }
    } else if (PySequence_Check(seq) && !PyBytes_Check(seq)) {
        long i, maxtype=BOOL_SCALAR, slen;

        slen = PySequence_Length(seq);
        if (slen < 0) return -1;

        if (slen == 0) return INT_SCALAR;

        for(i=0; i<slen; i++) {
            long newmax;
            PyObject *o = PySequence_GetItem(seq, i);
            if (!o) return -1;
            newmax = _NA_maxType(o, limit+1);
            if (newmax  < 0)
                return -1;
            else if (newmax > maxtype) {
                maxtype = newmax;
            }
            Py_DECREF(o);
        }
        return maxtype;
    } else {
        if (PyBool_Check(seq))
            return BOOL_SCALAR;
        else
#if defined(NPY_PY3K)
            if (PyInt_Check(seq))
                return INT_SCALAR;
            else if (PyLong_Check(seq))
#else
            if (PyLong_Check(seq))
#endif
                return LONG_SCALAR;
            else if (PyFloat_Check(seq))
                return FLOAT_SCALAR;
            else if (PyComplex_Check(seq))
                return COMPLEX_SCALAR;
            else {
                PyErr_Format(PyExc_TypeError,
                        "Expecting a python numeric type, got something else.");
                return -1;
            }
    }
}

static int
NA_maxType(PyObject *seq)
{
    int rval;
    rval = _NA_maxType(seq, 0);
    return rval;
}

static int
NA_isPythonScalar(PyObject *o)
{
    int rval;
    rval =  PyInt_Check(o) ||
        PyLong_Check(o) ||
        PyFloat_Check(o) ||
        PyComplex_Check(o) ||
        (PyBytes_Check(o) && (PyBytes_Size(o) == 1));
    return rval;
}

#if (NPY_SIZEOF_INTP == 8)
#define PlatBigInt PyInt_FromLong
#define PlatBigUInt PyLong_FromUnsignedLong
#else
#define PlatBigInt PyLong_FromLongLong
#define PlatBigUInt PyLong_FromUnsignedLongLong
#endif


static PyObject *
NA_getPythonScalar(PyArrayObject *a, long offset)
{
    int type = PyArray_DESCR(a)->type_num;
    PyObject *rval = NULL;

    switch(type) {
    case tBool:
    case tInt8:
    case tUInt8:
    case tInt16:
    case tUInt16:
    case tInt32: {
                     Int64 v = NA_get_Int64(a, offset);
                     rval = PyInt_FromLong(v);
                     break;
                 }
    case tUInt32: {
                      Int64 v = NA_get_Int64(a, offset);
                      rval = PlatBigUInt(v);
                      break;
                  }
    case tInt64: {
                     Int64 v = NA_get_Int64(a, offset);
                     rval = PlatBigInt( v);
                     break;
                 }
    case tUInt64: {
                      Int64 v = NA_get_Int64(a, offset);
                      rval = PlatBigUInt( v);
                      break;
                  }
    case tFloat32:
    case tFloat64: {
                       Float64 v = NA_get_Float64(a, offset);
                       rval = PyFloat_FromDouble( v );
                       break;
                   }
    case tComplex32:
    case tComplex64:
                   {
                       Complex64 v = NA_get_Complex64(a, offset);
                       rval = PyComplex_FromDoubles(v.r, v.i);
                       break;
                   }
    default:
                   rval = PyErr_Format(PyExc_TypeError,
                           "NA_getPythonScalar: bad type %d\n",
                           type);
    }
    return rval;
}

static int
NA_overflow(PyArrayObject *a, Float64 v)
{
    if ((PyArray_FLAGS(a) & CHECKOVERFLOW) == 0) return 0;

    switch(PyArray_DESCR(a)->type_num) {
    case tBool:
        return 0;
    case tInt8:
        if ((v < -128) || (v > 127))      goto _fail;
        return 0;
    case tUInt8:
        if ((v < 0) || (v > 255))         goto _fail;
        return 0;
    case tInt16:
        if ((v < -32768) || (v > 32767))  goto _fail;
        return 0;
    case tUInt16:
        if ((v < 0) || (v > 65535))       goto _fail;
        return 0;
    case tInt32:
        if ((v < -2147483648.) ||
                (v > 2147483647.))           goto _fail;
        return 0;
    case tUInt32:
        if ((v < 0) || (v > 4294967295.)) goto _fail;
        return 0;
    case tInt64:
        if ((v < -9223372036854775808.) ||
                (v > 9223372036854775807.))    goto _fail;
        return 0;
#if HAS_UINT64
    case tUInt64:
        if ((v < 0) ||
                (v > 18446744073709551615.))    goto _fail;
        return 0;
#endif
    case tFloat32:
        if ((v < -FLT_MAX) || (v > FLT_MAX)) goto _fail;
        return 0;
    case tFloat64:
        return 0;
    case tComplex32:
        if ((v < -FLT_MAX) || (v > FLT_MAX)) goto _fail;
        return 0;
    case tComplex64:
        return 0;
    default:
        PyErr_Format( PyExc_TypeError,
                "Unknown type %d in NA_overflow",
                PyArray_DESCR(a)->type_num );
        PyErr_Print();
        return -1;
    }
_fail:
    PyErr_Format(PyExc_OverflowError, "value out of range for array");
    return -1;
}

static int
_setFromPythonScalarCore(PyArrayObject *a, long offset, PyObject*value, int entries)
{
    Int64 v;
    if (entries >= 100) {
        PyErr_Format(PyExc_RuntimeError,
                "NA_setFromPythonScalar: __tonumtype__ conversion chain too long");
        return -1;
    } else if (PyInt_Check(value)) {
        v = PyInt_AsLong(value);
        if (NA_overflow(a, v) < 0)
            return -1;
        NA_set_Int64(a, offset, v);
    } else if (PyLong_Check(value)) {
        if (PyArray_DESCR(a)->type_num == tInt64) {
            v = (Int64) PyLong_AsLongLong( value );
        } else if (PyArray_DESCR(a)->type_num == tUInt64) {
            v = (UInt64) PyLong_AsUnsignedLongLong( value );
        } else if (PyArray_DESCR(a)->type_num == tUInt32) {
            v = PyLong_AsUnsignedLong(value);
        } else {
            v = PyLong_AsLongLong(value);
        }
        if (PyErr_Occurred())
            return -1;
        if (NA_overflow(a, v) < 0)
            return -1;
        NA_set_Int64(a, offset, v);
    } else if (PyFloat_Check(value)) {
        Float64 v = PyFloat_AsDouble(value);
        if (NA_overflow(a, v) < 0)
            return -1;
        NA_set_Float64(a, offset, v);
    } else if (PyComplex_Check(value)) {
        Complex64 vc;
        vc.r = PyComplex_RealAsDouble(value);
        vc.i = PyComplex_ImagAsDouble(value);
        if (NA_overflow(a, vc.r) < 0)
            return -1;
        if (NA_overflow(a, vc.i) < 0)
            return -1;
        NA_set_Complex64(a, offset, vc);
    } else if (PyObject_HasAttrString(value, "__tonumtype__")) {
        int rval;
        PyObject *type = NA_typeNoToTypeObject(PyArray_DESCR(a)->type_num);
        if (!type) return -1;
        value = PyObject_CallMethod(
                value, "__tonumtype__", "(N)", type);
        if (!value) return -1;
        rval = _setFromPythonScalarCore(a, offset, value, entries+1);
        Py_DECREF(value);
        return rval;
    } else if (PyBytes_Check(value)) {
        long size = PyBytes_Size(value);
        if ((size <= 0) || (size > 1)) {
            PyErr_Format( PyExc_ValueError,
                    "NA_setFromPythonScalar: len(string) must be 1.");
            return -1;
        }
        NA_set_Int64(a, offset, *PyBytes_AsString(value));
    } else {
        PyErr_Format(PyExc_TypeError,
                "NA_setFromPythonScalar: bad value type.");
        return -1;
    }
    return 0;
}

static int
NA_setFromPythonScalar(PyArrayObject *a, long offset, PyObject *value)
{
    if (PyArray_FailUnlessWriteable(a, "array") < 0) {
        return -1;
    }
    return _setFromPythonScalarCore(a, offset, value, 0);
}


static int
NA_NDArrayCheck(PyObject *obj) {
    return PyArray_Check(obj);
}

static int
NA_NumArrayCheck(PyObject *obj) {
    return PyArray_Check(obj);
}

static int
NA_ComplexArrayCheck(PyObject *a)
{
    int rval = NA_NumArrayCheck(a);
    if (rval > 0) {
        PyArrayObject *arr = (PyArrayObject *) a;
        switch(PyArray_DESCR(arr)->type_num) {
        case tComplex64: case tComplex32:
            return 1;
        default:
            return 0;
        }
    }
    return rval;
}

static unsigned long
NA_elements(PyArrayObject  *a)
{
    int i;
    unsigned long n = 1;
    for(i = 0; i<PyArray_NDIM(a); i++)
        n *= PyArray_DIMS(a)[i];
    return n;
}

static int
NA_typeObjectToTypeNo(PyObject *typeObj)
{
    PyArray_Descr *dtype;
    int i;
    if (PyArray_DescrConverter(typeObj, &dtype) == NPY_FAIL) i=-1;
    else i=dtype->type_num;
    return i;
}

static int
NA_copyArray(PyArrayObject *to, const PyArrayObject *from)
{
    return PyArray_CopyInto(to, (PyArrayObject *)from);
}

static PyArrayObject *
NA_copy(PyArrayObject *from)
{
    return (PyArrayObject *)PyArray_NewCopy(from, 0);
}


static PyObject *
NA_getType( PyObject *type)
{
    PyArray_Descr *typeobj = NULL;
    if (!type && PyArray_DescrConverter(type, &typeobj) == NPY_FAIL) {
        PyErr_Format(PyExc_ValueError, "NA_getType: unknown type.");
        typeobj = NULL;
    }
    return (PyObject *)typeobj;
}


/* Call a standard "stride" function
 **
 ** Stride functions always take one input and one output array.
 ** They can handle n-dimensional data with arbitrary strides (of
 ** either sign) for both the input and output numarray. Typically
 ** these functions are used to copy data, byteswap, or align data.
 **
 **
 ** It expects the following arguments:
 **
 **   Number of iterations for each dimension as a tuple
 **   Input Buffer Object
 **   Offset in bytes for input buffer
 **   Input strides (in bytes) for each dimension as a tuple
 **   Output Buffer Object
 **   Offset in bytes for output buffer
 **   Output strides (in bytes) for each dimension as a tuple
 **   An integer (Optional), typically the number of bytes to copy per
 *       element.
 **
 ** Returns None
 **
 ** The arguments expected by the standard stride functions that this
 ** function calls are:
 **
 **   Number of dimensions to iterate over
 **   Long int value (from the optional last argument to
 **      callStrideConvCFunc)
 **      often unused by the C Function
 **   An array of long ints. Each is the number of iterations for each
 **      dimension. NOTE: the previous argument as well as the stride
 **      arguments are reversed in order with respect to how they are
 **      used in Python. Fastest changing dimension is the first element
 **      in the numarray!
 **   A void pointer to the input data buffer.
 **   The starting offset for the input data buffer in bytes (long int).
 **   An array of long int input strides (in bytes) [reversed as with
 **      the iteration array]
 **   A void pointer to the output data buffer.
 **   The starting offset for the output data buffer in bytes (long int).
 **   An array of long int output strides (in bytes) [also reversed]
 */


static PyObject *
NA_callStrideConvCFuncCore(
        PyObject *self, int nshape, maybelong *shape,
        PyObject *inbuffObj,  long inboffset,
        int NPY_UNUSED(ninbstrides), maybelong *inbstrides,
        PyObject *outbuffObj, long outboffset,
        int NPY_UNUSED(noutbstrides), maybelong *outbstrides,
        long nbytes)
{
    CfuncObject *me = (CfuncObject *) self;
    CFUNC_STRIDE_CONV_FUNC funcptr;
    void *inbuffer, *outbuffer;
    long inbsize, outbsize;
    maybelong i, lshape[MAXDIM], in_strides[MAXDIM], out_strides[MAXDIM];
    maybelong shape_0, inbstr_0, outbstr_0;

    if (nshape == 0) {   /* handle rank-0 numarray. */
        nshape = 1;
        shape = &shape_0;
        inbstrides = &inbstr_0;
        outbstrides = &outbstr_0;
        shape[0] = 1;
        inbstrides[0] = outbstrides[0] = 0;
    }

    for(i=0; i<nshape; i++)
        lshape[i] = shape[nshape-1-i];
    for(i=0; i<nshape; i++)
        in_strides[i] = inbstrides[nshape-1-i];
    for(i=0; i<nshape; i++)
        out_strides[i] = outbstrides[nshape-1-i];

    if (!PyObject_IsInstance(self , (PyObject *) &CfuncType)
            || me->descr.type != CFUNC_STRIDING)
        return PyErr_Format(PyExc_TypeError,
                "NA_callStrideConvCFuncCore: problem with cfunc");

    if ((inbsize = NA_getBufferPtrAndSize(inbuffObj, 1, &inbuffer)) < 0)
        return PyErr_Format(_Error,
                "%s: Problem with input buffer", me->descr.name);

    if ((outbsize = NA_getBufferPtrAndSize(outbuffObj, 0, &outbuffer)) < 0)
        return PyErr_Format(_Error,
                "%s: Problem with output buffer (read only?)",
                me->descr.name);

    /* Check buffer alignment and bounds */
    if (NA_checkOneStriding(me->descr.name, nshape, lshape,
                inboffset, in_strides, inbsize,
                (me->descr.sizes[0] == -1) ?
                nbytes : me->descr.sizes[0],
                me->descr.align) ||
            NA_checkOneStriding(me->descr.name, nshape, lshape,
                outboffset, out_strides, outbsize,
                (me->descr.sizes[1] == -1) ?
                nbytes : me->descr.sizes[1],
                me->descr.align))
        return NULL;

    /* Cast function pointer and perform stride operation */
    funcptr = (CFUNC_STRIDE_CONV_FUNC) me->descr.fptr;
    if ((*funcptr)(nshape-1, nbytes, lshape,
                inbuffer,  inboffset, in_strides,
                outbuffer, outboffset, out_strides) == 0) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        return NULL;
    }
}

static void
NA_stridesFromShape(int nshape, maybelong *shape, maybelong bytestride,
        maybelong *strides)
{
    int i;
    if (nshape > 0) {
        for(i=0; i<nshape; i++)
            strides[i] = bytestride;
        for(i=nshape-2; i>=0; i--)
            strides[i] = strides[i+1]*shape[i+1];
    }
}

static int
NA_OperatorCheck(PyObject *NPY_UNUSED(op)) {
    return 0;
}

static int
NA_ConverterCheck(PyObject *NPY_UNUSED(op)) {
    return 0;
}

static int
NA_UfuncCheck(PyObject *NPY_UNUSED(op)) {
    return 0;
}

static int
NA_CfuncCheck(PyObject *op) {
    return PyObject_TypeCheck(op, &CfuncType);
}

static int
NA_getByteOffset(PyArrayObject *NPY_UNUSED(array), int NPY_UNUSED(nindices),
        maybelong *NPY_UNUSED(indices), long *NPY_UNUSED(offset))
{
    return 0;
}

static int
NA_swapAxes(PyArrayObject *array, int x, int y)
{
    long temp;

    if (((PyObject *) array) == Py_None) return 0;

    if (PyArray_NDIM(array) < 2) return 0;

    if (x < 0) x += PyArray_NDIM(array);
    if (y < 0) y += PyArray_NDIM(array);

    if ((x < 0) || (x >= PyArray_NDIM(array)) ||
            (y < 0) || (y >= PyArray_NDIM(array))) {
        PyErr_Format(PyExc_ValueError,
                "Specified dimension does not exist");
        return -1;
    }

    temp = PyArray_DIMS(array)[x];
    PyArray_DIMS(array)[x] = PyArray_DIMS(array)[y];
    PyArray_DIMS(array)[y] = temp;

    temp = PyArray_STRIDES(array)[x];
    PyArray_STRIDES(array)[x] = PyArray_STRIDES(array)[y];
    PyArray_STRIDES(array)[y] = temp;

    PyArray_UpdateFlags(array, NPY_ARRAY_UPDATE_ALL);

    return 0;
}

static PyObject *
NA_initModuleGlobal(char *modulename, char *globalname)
{
    PyObject *module, *dict, *global = NULL;
    module = PyImport_ImportModule(modulename);
    if (!module) {
        PyErr_Format(PyExc_RuntimeError,
                "Can't import '%s' module",
                modulename);
        goto _exit;
    }
    dict = PyModule_GetDict(module);
    global = PyDict_GetItemString(dict, globalname);
    if (!global) {
        PyErr_Format(PyExc_RuntimeError,
                "Can't find '%s' global in '%s' module.",
                globalname, modulename);
        goto _exit;
    }
    Py_DECREF(module);
    Py_INCREF(global);
_exit:
    return global;
}

    NumarrayType
NA_NumarrayType(PyObject *seq)
{
    int maxtype = NA_maxType(seq);
    int rval;
    switch(maxtype) {
    case BOOL_SCALAR:
        rval = tBool;
        goto _exit;
    case INT_SCALAR:
    case LONG_SCALAR:
        rval = tLong; /* tLong corresponds to C long int,
                         not Python long int */
        goto _exit;
    case FLOAT_SCALAR:
        rval = tFloat64;
        goto _exit;
    case COMPLEX_SCALAR:
        rval = tComplex64;
        goto _exit;
    default:
        PyErr_Format(PyExc_TypeError,
                "expecting Python numeric scalar value; got something else.");
        rval = -1;
    }
_exit:
    return rval;
}

/* ignores bytestride */
static PyArrayObject *
NA_NewAllFromBuffer(int ndim, maybelong *shape, NumarrayType type,
        PyObject *bufferObject, maybelong byteoffset,
        maybelong NPY_UNUSED(bytestride), int byteorder,
        int NPY_UNUSED(aligned), int NPY_UNUSED(writeable))
{
    PyArrayObject *self = NULL;
    PyArray_Descr *dtype;

    if (type == tAny)
        type = tDefault;

    dtype = PyArray_DescrFromType(type);
    if (dtype == NULL) return NULL;

    if (byteorder != NA_ByteOrder()) {
        PyArray_Descr *temp;
        temp = PyArray_DescrNewByteorder(dtype, NPY_SWAP);
        Py_DECREF(dtype);
        if (temp == NULL) return NULL;
        dtype = temp;
    }

    if (bufferObject == Py_None || bufferObject == NULL) {
        self = (PyArrayObject *)        \
               PyArray_NewFromDescr(&PyArray_Type, dtype,
                       ndim, shape, NULL, NULL,
                       0, NULL);
    }
    else {
        npy_intp size = 1;
        int i;
        PyArrayObject *newself;
        PyArray_Dims newdims;
        for(i=0; i<ndim; i++) {
            size *= shape[i];
        }
        self = (PyArrayObject *)\
               PyArray_FromBuffer(bufferObject, dtype,
                       size, byteoffset);

        if (self == NULL) return self;
        newdims.len = ndim;
        newdims.ptr = shape;
        newself = (PyArrayObject *)\
                  PyArray_Newshape(self, &newdims, NPY_CORDER);
        Py_DECREF(self);
        self = newself;
    }

    return self;
}

static void
NA_updateAlignment(PyArrayObject *self)
{
    PyArray_UpdateFlags(self, NPY_ARRAY_ALIGNED);
}

static void
NA_updateContiguous(PyArrayObject *self)
{
    PyArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
}


static void
NA_updateStatus(PyArrayObject *self)
{
    PyArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);
}

static int
NA_NumArrayCheckExact(PyObject *op) {
    return (op->ob_type == &PyArray_Type);
}

static int
NA_NDArrayCheckExact(PyObject *op) {
    return (op->ob_type == &PyArray_Type);
}

static int
NA_OperatorCheckExact(PyObject *NPY_UNUSED(op)) {
    return 0;
}

static int
NA_ConverterCheckExact(PyObject *NPY_UNUSED(op)) {
    return 0;
}

static int
NA_UfuncCheckExact(PyObject *NPY_UNUSED(op)) {
    return 0;
}


static int
NA_CfuncCheckExact(PyObject *op) {
    return op->ob_type == &CfuncType;
}

static char *
NA_getArrayData(PyArrayObject *obj)
{
    if (!NA_NDArrayCheck((PyObject *) obj)) {
        PyErr_Format(PyExc_TypeError,
                "expected an NDArray");
    }
    return PyArray_DATA(obj);
}

/* Byteswap is not a flag of the array --- it is implicit in the data-type */
static void
NA_updateByteswap(PyArrayObject *NPY_UNUSED(self))
{
    return;
}

static PyArray_Descr *
NA_DescrFromType(int type)
{
    if (type == tAny)
        type = tDefault;
    return PyArray_DescrFromType(type);
}

static PyObject *
NA_Cast(PyArrayObject *a, int type)
{
    return PyArray_Cast(a, type);
}


/* The following function has much platform dependent code since
 ** there is no platform-independent way of checking Floating Point
 ** status bits
 */

/*  OSF/Alpha (Tru64)  ---------------------------------------------*/
#if defined(__osf__) && defined(__alpha)

static int
NA_checkFPErrors(void)
{
    unsigned long fpstatus;
    int retstatus;

#include <machine/fpu.h>   /* Should migrate to global scope */

    fpstatus = ieee_get_fp_control();
    /* clear status bits as well as disable exception mode if on */
    ieee_set_fp_control( 0 );
    retstatus =
        pyFPE_DIVIDE_BY_ZERO* (int)((IEEE_STATUS_DZE  & fpstatus) != 0)
        + pyFPE_OVERFLOW      * (int)((IEEE_STATUS_OVF  & fpstatus) != 0)
        + pyFPE_UNDERFLOW     * (int)((IEEE_STATUS_UNF  & fpstatus) != 0)
        + pyFPE_INVALID       * (int)((IEEE_STATUS_INV  & fpstatus) != 0);

    return retstatus;
}

/* MS Windows -----------------------------------------------------*/
#elif defined(_MSC_VER)

#include <float.h>

static int
NA_checkFPErrors(void)
{
    int fpstatus = (int) _clear87();
    int retstatus =
        pyFPE_DIVIDE_BY_ZERO * ((SW_ZERODIVIDE & fpstatus) != 0)
        + pyFPE_OVERFLOW       * ((SW_OVERFLOW & fpstatus)   != 0)
        + pyFPE_UNDERFLOW      * ((SW_UNDERFLOW & fpstatus)  != 0)
        + pyFPE_INVALID        * ((SW_INVALID & fpstatus)    != 0);


    return retstatus;
}

/* Solaris --------------------------------------------------------*/
/* --------ignoring SunOS ieee_flags approach, someone else can
 **         deal with that! */
#elif defined(sun)
#include <ieeefp.h>

static int
NA_checkFPErrors(void)
{
    int fpstatus;
    int retstatus;

    fpstatus = (int) fpgetsticky();
    retstatus = pyFPE_DIVIDE_BY_ZERO * ((FP_X_DZ  & fpstatus) != 0)
        + pyFPE_OVERFLOW       * ((FP_X_OFL & fpstatus) != 0)
        + pyFPE_UNDERFLOW      * ((FP_X_UFL & fpstatus) != 0)
        + pyFPE_INVALID        * ((FP_X_INV & fpstatus) != 0);
    (void) fpsetsticky(0);

    return retstatus;
}

#elif defined(__GLIBC__) || defined(__APPLE__) || defined(__CYGWIN__) || defined(__MINGW32__) || (defined(__FreeBSD__) && (__FreeBSD_version >= 502114))

static int
NA_checkFPErrors(void)
{
    int fpstatus = (int) fetestexcept(
            FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    int retstatus =
        pyFPE_DIVIDE_BY_ZERO * ((FE_DIVBYZERO  & fpstatus) != 0)
        + pyFPE_OVERFLOW       * ((FE_OVERFLOW   & fpstatus) != 0)
        + pyFPE_UNDERFLOW      * ((FE_UNDERFLOW  & fpstatus) != 0)
        + pyFPE_INVALID        * ((FE_INVALID    & fpstatus) != 0);
    (void) feclearexcept(FE_DIVBYZERO | FE_OVERFLOW |
            FE_UNDERFLOW | FE_INVALID);
    return retstatus;
}

#else

static int
NA_checkFPErrors(void)
{
    return 0;
}

#endif

static void
NA_clearFPErrors()
{
    NA_checkFPErrors();
}

/* Not supported yet */
static int
NA_checkAndReportFPErrors(char *name)
{
    int error = NA_checkFPErrors();
    if (error) {
        PyObject *ans;
        char msg[128];
        strcpy(msg, " in ");
        strncat(msg, name, 100);
        ans = PyObject_CallFunction(pHandleErrorFunc, "(is)", error, msg);
        if (!ans) return -1;
        Py_DECREF(ans); /* Py_None */
    }
    return 0;

}


#define WITHIN32(v, f) (((v) >= f##_MIN32) && ((v) <= f##_MAX32))
#define WITHIN64(v, f) (((v) >= f##_MIN64) && ((v) <= f##_MAX64))

static Bool
NA_IeeeMask32( Float32 f, Int32 mask)
{
    Int32 category;
    UInt32 v = *(UInt32 *) &f;

    if (v & BIT(31)) {
        if (WITHIN32(v, NEG_NORMALIZED)) {
            category = MSK_NEG_NOR;
        } else if (WITHIN32(v, NEG_DENORMALIZED)) {
            category = MSK_NEG_DEN;
        } else if (WITHIN32(v, NEG_SIGNAL_NAN)) {
            category = MSK_NEG_SNAN;
        } else if (WITHIN32(v, NEG_QUIET_NAN)) {
            category = MSK_NEG_QNAN;
        } else if (v == NEG_INFINITY_MIN32) {
            category = MSK_NEG_INF;
        } else if (v == NEG_ZERO_MIN32) {
            category = MSK_NEG_ZERO;
        } else if (v == INDETERMINATE_MIN32) {
            category = MSK_INDETERM;
        } else {
            category = MSK_BUG;
        }
    } else {
        if (WITHIN32(v, POS_NORMALIZED)) {
            category = MSK_POS_NOR;
        } else if (WITHIN32(v, POS_DENORMALIZED)) {
            category = MSK_POS_DEN;
        } else if (WITHIN32(v, POS_SIGNAL_NAN)) {
            category = MSK_POS_SNAN;
        } else if (WITHIN32(v, POS_QUIET_NAN)) {
            category = MSK_POS_QNAN;
        } else if (v == POS_INFINITY_MIN32) {
            category = MSK_POS_INF;
        } else if (v == POS_ZERO_MIN32) {
            category = MSK_POS_ZERO;
        } else {
            category = MSK_BUG;
        }
    }
    return (category & mask) != 0;
}

static Bool
NA_IeeeMask64( Float64 f, Int32 mask)
{
    Int32 category;
    UInt64 v = *(UInt64 *) &f;

    if (v & BIT(63)) {
        if (WITHIN64(v, NEG_NORMALIZED)) {
            category = MSK_NEG_NOR;
        } else if (WITHIN64(v, NEG_DENORMALIZED)) {
            category = MSK_NEG_DEN;
        } else if (WITHIN64(v, NEG_SIGNAL_NAN)) {
            category = MSK_NEG_SNAN;
        } else if (WITHIN64(v, NEG_QUIET_NAN)) {
            category = MSK_NEG_QNAN;
        } else if (v == NEG_INFINITY_MIN64) {
            category = MSK_NEG_INF;
        } else if (v == NEG_ZERO_MIN64) {
            category = MSK_NEG_ZERO;
        } else if (v == INDETERMINATE_MIN64) {
            category = MSK_INDETERM;
        } else {
            category = MSK_BUG;
        }
    } else {
        if (WITHIN64(v, POS_NORMALIZED)) {
            category = MSK_POS_NOR;
        } else if (WITHIN64(v, POS_DENORMALIZED)) {
            category = MSK_POS_DEN;
        } else if (WITHIN64(v, POS_SIGNAL_NAN)) {
            category = MSK_POS_SNAN;
        } else if (WITHIN64(v, POS_QUIET_NAN)) {
            category = MSK_POS_QNAN;
        } else if (v == POS_INFINITY_MIN64) {
            category = MSK_POS_INF;
        } else if (v == POS_ZERO_MIN64) {
            category = MSK_POS_ZERO;
        } else {
            category = MSK_BUG;
        }
    }
    return (category & mask) != 0;
}

static PyArrayObject *
NA_FromDimsStridesDescrAndData(int nd, maybelong *d, maybelong *s, PyArray_Descr *descr, char *data)
{
    return (PyArrayObject *)\
        PyArray_NewFromDescr(&PyArray_Type, descr, nd, d,
                s, data, 0, NULL);
}

static PyArrayObject *
NA_FromDimsTypeAndData(int nd, maybelong *d, int type, char *data)
{
    PyArray_Descr *descr = NA_DescrFromType(type);
    return NA_FromDimsStridesDescrAndData(nd, d, NULL, descr, data);
}

static PyArrayObject *
NA_FromDimsStridesTypeAndData(int nd, maybelong *shape, maybelong *strides,
        int type, char *data)
{
    PyArray_Descr *descr = NA_DescrFromType(type);
    return NA_FromDimsStridesDescrAndData(nd, shape, strides, descr, data);
}


typedef struct
{
    NumarrayType type_num;
    char suffix[5];
    int  itemsize;
} scipy_typestr;

static scipy_typestr scipy_descriptors[ ] = {
    { tAny,       "", 0},

    { tBool,      "b1", 1},

    { tInt8,      "i1", 1},
    { tUInt8,     "u1", 1},

    { tInt16,     "i2", 2},
    { tUInt16,    "u2", 2},

    { tInt32,     "i4", 4},
    { tUInt32,    "u4", 4},

    { tInt64,     "i8", 8},
    { tUInt64,    "u8", 8},

    { tFloat32,   "f4", 4},
    { tFloat64,   "f8", 8},

    { tComplex32, "c8", 8},
    { tComplex64, "c16", 16}
};


static int
NA_scipy_typestr(NumarrayType t, int byteorder, char *typestr)
{
    size_t i;
    if (byteorder)
        strcpy(typestr, ">");
    else
        strcpy(typestr, "<");
    for(i=0; i<ELEM(scipy_descriptors); i++) {
        scipy_typestr *ts = &scipy_descriptors[i];
        if (ts->type_num == t) {
            strncat(typestr, ts->suffix, 4);
            return 0;
        }
    }
    return -1;
}

static PyArrayObject *
NA_FromArrayStruct(PyObject *obj)
{
    return (PyArrayObject *)PyArray_FromStructInterface(obj);
}


static PyObject *_Error;

void *libnumarray_API[] = {
    (void*) getBuffer,
    (void*) isBuffer,
    (void*) getWriteBufferDataPtr,
    (void*) isBufferWriteable,
    (void*) getReadBufferDataPtr,
    (void*) getBufferSize,
    (void*) num_log,
    (void*) num_log10,
    (void*) num_pow,
    (void*) num_acosh,
    (void*) num_asinh,
    (void*) num_atanh,
    (void*) num_round,
    (void*) int_dividebyzero_error,
    (void*) int_overflow_error,
    (void*) umult64_overflow,
    (void*) smult64_overflow,
    (void*) NA_Done,
    (void*) NA_NewAll,
    (void*) NA_NewAllStrides,
    (void*) NA_New,
    (void*) NA_Empty,
    (void*) NA_NewArray,
    (void*) NA_vNewArray,
    (void*) NA_ReturnOutput,
    (void*) NA_getBufferPtrAndSize,
    (void*) NA_checkIo,
    (void*) NA_checkOneCBuffer,
    (void*) NA_checkNCBuffers,
    (void*) NA_checkOneStriding,
    (void*) NA_new_cfunc,
    (void*) NA_add_cfunc,
    (void*) NA_InputArray,
    (void*) NA_OutputArray,
    (void*) NA_IoArray,
    (void*) NA_OptionalOutputArray,
    (void*) NA_get_offset,
    (void*) NA_get_Float64,
    (void*) NA_set_Float64,
    (void*) NA_get_Complex64,
    (void*) NA_set_Complex64,
    (void*) NA_get_Int64,
    (void*) NA_set_Int64,
    (void*) NA_get1_Float64,
    (void*) NA_get2_Float64,
    (void*) NA_get3_Float64,
    (void*) NA_set1_Float64,
    (void*) NA_set2_Float64,
    (void*) NA_set3_Float64,
    (void*) NA_get1_Complex64,
    (void*) NA_get2_Complex64,
    (void*) NA_get3_Complex64,
    (void*) NA_set1_Complex64,
    (void*) NA_set2_Complex64,
    (void*) NA_set3_Complex64,
    (void*) NA_get1_Int64,
    (void*) NA_get2_Int64,
    (void*) NA_get3_Int64,
    (void*) NA_set1_Int64,
    (void*) NA_set2_Int64,
    (void*) NA_set3_Int64,
    (void*) NA_get1D_Float64,
    (void*) NA_set1D_Float64,
    (void*) NA_get1D_Int64,
    (void*) NA_set1D_Int64,
    (void*) NA_get1D_Complex64,
    (void*) NA_set1D_Complex64,
    (void*) NA_ShapeEqual,
    (void*) NA_ShapeLessThan,
    (void*) NA_ByteOrder,
    (void*) NA_IeeeSpecial32,
    (void*) NA_IeeeSpecial64,
    (void*) NA_updateDataPtr,
    (void*) NA_typeNoToName,
    (void*) NA_nameToTypeNo,
    (void*) NA_typeNoToTypeObject,
    (void*) NA_intTupleFromMaybeLongs,
    (void*) NA_maybeLongsFromIntTuple,
    (void*) NA_intTupleProduct,
    (void*) NA_isIntegerSequence,
    (void*) NA_setArrayFromSequence,
    (void*) NA_maxType,
    (void*) NA_isPythonScalar,
    (void*) NA_getPythonScalar,
    (void*) NA_setFromPythonScalar,
    (void*) NA_NDArrayCheck,
    (void*) NA_NumArrayCheck,
    (void*) NA_ComplexArrayCheck,
    (void*) NA_elements,
    (void*) NA_typeObjectToTypeNo,
    (void*) NA_copyArray,
    (void*) NA_copy,
    (void*) NA_getType,
    (void*) NA_callCUFuncCore,
    (void*) NA_callStrideConvCFuncCore,
    (void*) NA_stridesFromShape,
    (void*) NA_OperatorCheck,
    (void*) NA_ConverterCheck,
    (void*) NA_UfuncCheck,
    (void*) NA_CfuncCheck,
    (void*) NA_getByteOffset,
    (void*) NA_swapAxes,
    (void*) NA_initModuleGlobal,
    (void*) NA_NumarrayType,
    (void*) NA_NewAllFromBuffer,
    (void*) NA_alloc1D_Float64,
    (void*) NA_alloc1D_Int64,
    (void*) NA_updateAlignment,
    (void*) NA_updateContiguous,
    (void*) NA_updateStatus,
    (void*) NA_NumArrayCheckExact,
    (void*) NA_NDArrayCheckExact,
    (void*) NA_OperatorCheckExact,
    (void*) NA_ConverterCheckExact,
    (void*) NA_UfuncCheckExact,
    (void*) NA_CfuncCheckExact,
    (void*) NA_getArrayData,
    (void*) NA_updateByteswap,
    (void*) NA_DescrFromType,
    (void*) NA_Cast,
    (void*) NA_checkFPErrors,
    (void*) NA_clearFPErrors,
    (void*) NA_checkAndReportFPErrors,
    (void*) NA_IeeeMask32,
    (void*) NA_IeeeMask64,
    (void*) _NA_callStridingHelper,
    (void*) NA_FromDimsStridesDescrAndData,
    (void*) NA_FromDimsTypeAndData,
    (void*) NA_FromDimsStridesTypeAndData,
    (void*) NA_scipy_typestr,
    (void*) NA_FromArrayStruct
};

#if (!defined(METHOD_TABLE_EXISTS))
static PyMethodDef _libnumarrayMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
#endif

/* boiler plate API init */
#if defined(NPY_PY3K)

#define RETVAL m

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_capi",
        NULL,
        -1,
        _libnumarrayMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__capi(void)
#else

#define RETVAL

PyMODINIT_FUNC init_capi(void)
#endif
{
    PyObject *m;
    PyObject *c_api_object;

    _Error = PyErr_NewException("numpy.numarray._capi.error", NULL, NULL);

    /* Create a CObject containing the API pointer array's address */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_capi", _libnumarrayMethods);
#endif

#if defined(NPY_PY3K)
    c_api_object = PyCapsule_New((void *)libnumarray_API, NULL, NULL);
    if (c_api_object == NULL) {
        PyErr_Clear();
    }
#else
    c_api_object = PyCObject_FromVoidPtr((void *)libnumarray_API, NULL);
#endif

    if (c_api_object != NULL) {
        /* Create a name for this object in the module's namespace */
        PyObject *d = PyModule_GetDict(m);

        PyDict_SetItemString(d, "_C_API", c_api_object);
        PyDict_SetItemString(d, "error", _Error);
        Py_DECREF(c_api_object);
    }
    else {
        return RETVAL;
    }
    if (PyModule_AddObject(m, "__version__", PyUString_FromString("0.9")) < 0) {
        return RETVAL;
    }
    if (_import_array() < 0) {
        return RETVAL;
    }
    deferred_libnumarray_init();
    return RETVAL;
}

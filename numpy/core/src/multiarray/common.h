#ifndef _NPY_PRIVATE_COMMON_H_
#define _NPY_PRIVATE_COMMON_H_
#include "structmember.h"
#include <numpy/npy_common.h>
#include <numpy/npy_cpu.h>
#include <numpy/ndarraytypes.h>
#include <limits.h>

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

#ifdef NPY_ALLOW_THREADS
#define NPY_BEGIN_THREADS_NDITER(iter) \
        do { \
            if (!NpyIter_IterationNeedsAPI(iter)) { \
                NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter)); \
            } \
        } while(0)
#else
#define NPY_BEGIN_THREADS_NDITER(iter)
#endif

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims,
                        PyArray_Descr **out_dtype);

NPY_NO_EXPORT int
PyArray_DTypeFromObjectHelper(PyObject *obj, int maxdims,
                              PyArray_Descr **out_dtype, int string_status);

/*
 * Returns NULL without setting an exception if no scalar is matched, a
 * new dtype reference otherwise.
 */
NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *str);

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, npy_intp i);

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret);

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap);

NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending);

/*
 * Sets ValueError with "matrices not aligned" message for np.dot and friends
 * when a.shape[i] should match b.shape[j], but doesn't.
 */
NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j);

/**
 * unpack tuple of dtype->fields (descr, offset, title[not-needed])
 *
 * @param "value" should be the tuple.
 *
 * @return "descr" will be set to the field's dtype
 * @return "offset" will be set to the field's offset
 *
 * returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset);

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype);

/*
 * Returns -1 and sets an exception if *index is an invalid index for
 * an array of size max_item, otherwise adjusts it in place to be
 * 0 <= *index < max_item, and returns 0.
 * 'axis' should be the array axis that is being indexed over, if known. If
 * unknown, use -1.
 * If _save is NULL it is assumed the GIL is taken
 * If _save is not NULL it is assumed the GIL is not taken and it
 * is acquired in the case of an error
 */
static NPY_INLINE int
check_and_adjust_index(npy_intp *index, npy_intp max_item, int axis,
                       PyThreadState * _save)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*index < -max_item) || (*index >= max_item))) {
        NPY_END_THREADS;
        /* Try to be as clear as possible about what went wrong. */
        if (axis >= 0) {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for axis %d with size %"NPY_INTP_FMT,
                         *index, axis, max_item);
        } else {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for size %"NPY_INTP_FMT, *index, max_item);
        }
        return -1;
    }
    /* adjust negative indices */
    if (*index < 0) {
        *index += max_item;
    }
    return 0;
}

/*
 * Returns -1 and sets an exception if *axis is an invalid axis for
 * an array of dimension ndim, otherwise adjusts it in place to be
 * 0 <= *axis < ndim, and returns 0.
 *
 * msg_prefix: borrowed reference, a string to prepend to the message
 */
static NPY_INLINE int
check_and_adjust_axis_msg(int *axis, int ndim, PyObject *msg_prefix)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        /*
         * Load the exception type, if we don't already have it. Unfortunately
         * we don't have access to npy_cache_import here
         */
        static PyObject *AxisError_cls = NULL;
        PyObject *exc;

        if (AxisError_cls == NULL) {
            PyObject *mod = PyImport_ImportModule("numpy.core._internal");

            if (mod != NULL) {
                AxisError_cls = PyObject_GetAttrString(mod, "AxisError");
                Py_DECREF(mod);
            }
        }

        /* Invoke the AxisError constructor */
        exc = PyObject_CallFunction(AxisError_cls, "iiO",
                                    *axis, ndim, msg_prefix);
        if (exc == NULL) {
            return -1;
        }
        PyErr_SetObject(AxisError_cls, exc);
        Py_DECREF(exc);

        return -1;
    }
    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;
    }
    return 0;
}
static NPY_INLINE int
check_and_adjust_axis(int *axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim, Py_None);
}

/* used for some alignment checks */
#define _ALIGN(type) offsetof(struct {char c; type v;}, v)
#define _UINT_ALIGN(type) npy_uint_alignment(sizeof(type))
/*
 * Disable harmless compiler warning "4116: unnamed type definition in
 * parentheses" which is caused by the _ALIGN macro.
 */
#if defined(_MSC_VER)
#pragma warning(disable:4116)
#endif

/*
 * return true if pointer is aligned to 'alignment'
 */
static NPY_INLINE int
npy_is_aligned(const void * p, const npy_uintp alignment)
{
    /*
     * Assumes alignment is a power of two, as required by the C standard.
     * Assumes cast from pointer to uintp gives a sensible representation we
     * can use bitwise & on (not required by C standard, but used by glibc).
     * This test is faster than a direct modulo.
     * Note alignment value of 0 is allowed and returns False.
     */
    return ((npy_uintp)(p) & ((alignment) - 1)) == 0;
}

/* Get equivalent "uint" alignment given an itemsize, for use in copy code */
static NPY_INLINE int
npy_uint_alignment(int itemsize)
{
    npy_uintp alignment = 0; /* return value of 0 means unaligned */

    switch(itemsize){
        case 1:
            return 1;
        case 2:
            alignment = _ALIGN(npy_uint16);
            break;
        case 4:
            alignment = _ALIGN(npy_uint32);
            break;
        case 8:
            alignment = _ALIGN(npy_uint64);
            break;
        case 16:
            /*
             * 16 byte types are copied using 2 uint64 assignments.
             * See the strided copy function in lowlevel_strided_loops.c.
             */
            alignment = _ALIGN(npy_uint64);
            break;
        default:
            break;
    }
    
    return alignment;
}

/*
 * memchr with stride and invert argument
 * intended for small searches where a call out to libc memchr is costly.
 * stride must be a multiple of size.
 * compared to memchr it returns one stride past end instead of NULL if needle
 * is not found.
 */
static NPY_INLINE char *
npy_memchr(char * haystack, char needle,
           npy_intp stride, npy_intp size, npy_intp * psubloopsize, int invert)
{
    char * p = haystack;
    npy_intp subloopsize = 0;

    if (!invert) {
        /*
         * this is usually the path to determine elements to process,
         * performance less important here.
         * memchr has large setup cost if 0 byte is close to start.
         */
        while (subloopsize < size && *p != needle) {
            subloopsize++;
            p += stride;
        }
    }
    else {
        /* usually find elements to skip path */
        if (NPY_CPU_HAVE_UNALIGNED_ACCESS && needle == 0 && stride == 1) {
            /* iterate until last multiple of 4 */
            char * block_end = haystack + size - (size % sizeof(unsigned int));
            while (p < block_end) {
                unsigned int  v = *(unsigned int*)p;
                if (v != 0) {
                    break;
                }
                p += sizeof(unsigned int);
            }
            /* handle rest */
            subloopsize = (p - haystack);
        }
        while (subloopsize < size && *p == needle) {
            subloopsize++;
            p += stride;
        }
    }

    *psubloopsize = subloopsize;

    return p;
}

/*
 * Convert NumPy stride to BLAS stride. Returns 0 if conversion cannot be done
 * (BLAS won't handle negative or zero strides the way we want).
 */
static NPY_INLINE int
blas_stride(npy_intp stride, unsigned itemsize)
{
    /*
     * Should probably check pointer alignment also, but this may cause
     * problems if we require complex to be 16 byte aligned.
     */
    if (stride > 0 && npy_is_aligned((void *)stride, itemsize)) {
        stride /= itemsize;
        if (stride <= INT_MAX) {
            return stride;
        }
    }
    return 0;
}

/*
 * Define a chunksize for CBLAS. CBLAS counts in integers.
 */
#if NPY_MAX_INTP > INT_MAX
# define NPY_CBLAS_CHUNK  (INT_MAX / 2 + 1)
#else
# define NPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif

#include "ucsnarrow.h"

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result);

#endif

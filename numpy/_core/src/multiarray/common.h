#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_

#include <structmember.h>
#include "numpy/npy_common.h"
#include "numpy/ndarraytypes.h"
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h"
#include "numpy/npy_cpu.h"

#include "npy_import.h"
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


NPY_NO_EXPORT PyArray_Descr *
PyArray_DTypeFromObjectStringDiscovery(
        PyObject *obj, PyArray_Descr *last_dtype, int string_type);

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


/*
 * Returns NULL without setting an exception if no scalar is matched, a
 * new dtype reference otherwise.
 */
NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap);

NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending);

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
static inline int
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
static inline int
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

        npy_cache_import("numpy.exceptions", "AxisError", &AxisError_cls);
        if (AxisError_cls == NULL) {
            return -1;
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
static inline int
check_and_adjust_axis(int *axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim, Py_None);
}

/* used for some alignment checks */
/*
 * GCC releases before GCC 4.9 had a bug in _Alignof.  See GCC bug 52023
 * <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52023>.
 * clang versions < 8.0.0 have the same bug.
 */
#if (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112 \
     || (defined __GNUC__ && __GNUC__ < 4 + (__GNUC_MINOR__ < 9) \
  && !defined __clang__) \
     || (defined __clang__ && __clang_major__ < 8))
# define NPY_ALIGNOF(type) offsetof(struct {char c; type v;}, v)
#else
# define NPY_ALIGNOF(type) _Alignof(type)
#endif
#define  NPY_ALIGNOF_UINT(type) npy_uint_alignment(sizeof(type))
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
static inline int
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
static inline npy_uintp
npy_uint_alignment(int itemsize)
{
    npy_uintp alignment = 0; /* return value of 0 means unaligned */

    switch(itemsize){
        case 1:
            return 1;
        case 2:
            alignment = NPY_ALIGNOF(npy_uint16);
            break;
        case 4:
            alignment = NPY_ALIGNOF(npy_uint32);
            break;
        case 8:
            alignment = NPY_ALIGNOF(npy_uint64);
            break;
        case 16:
            /*
             * 16 byte types are copied using 2 uint64 assignments.
             * See the strided copy function in lowlevel_strided_loops.c.
             */
            alignment = NPY_ALIGNOF(npy_uint64);
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
#ifdef __clang__
    /*
     * The code below currently makes use of !NPY_ALIGNMENT_REQUIRED, which
     * should be OK but causes the clang sanitizer to warn.  It may make
     * sense to modify the code to avoid this "unaligned" access but
     * it would be good to carefully check the performance changes.
     */
    __attribute__((no_sanitize("alignment")))
#endif
static inline char *
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
        if (!NPY_ALIGNMENT_REQUIRED && needle == 0 && stride == 1) {
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
 * Simple helper to create a tuple from an array of items. The `make_null_none`
 * flag means that NULL entries are replaced with None, which is occasionally
 * useful.
 */
static inline PyObject *
PyArray_TupleFromItems(int n, PyObject *const *items, int make_null_none)
{
    PyObject *tuple = PyTuple_New(n);
    if (tuple == NULL) {
        return NULL;
    }
    for (int i = 0; i < n; i ++) {
        PyObject *tmp;
        if (!make_null_none || items[i] != NULL) {
            tmp = items[i];
        }
        else {
            tmp = Py_None;
        }
        Py_INCREF(tmp);
        PyTuple_SET_ITEM(tuple, i, tmp);
    }
    return tuple;
}

/*
 * Returns 0 if the array has rank 0, -1 otherwise. Prints a deprecation
 * warning for arrays of _size_ 1.
 */
NPY_NO_EXPORT int
check_is_convertible_to_scalar(PyArrayObject *v);


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


/*
 * Used to indicate a broadcast axis, see also `npyiter_get_op_axis` in
 * `nditer_constr.c`.  This may be the preferred API for reduction axes
 * probably. So we should consider making this public either as a macro or
 * function (so that the way we flag the axis can be changed).
 */
#define NPY_ITER_REDUCTION_AXIS(axis) (axis + (1 << (NPY_BITSOF_INT - 2)))

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_ */

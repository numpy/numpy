#ifndef _NPY_PRIVATE_COMMON_H_
#define _NPY_PRIVATE_COMMON_H_
#include <numpy/npy_common.h>
#include <numpy/npy_cpu.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

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

NPY_NO_EXPORT PyObject *
PyArray_GetAttrString_SuppressException(PyObject *v, char *name);

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

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap);

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap);

NPY_NO_EXPORT void
offset_bounds_from_strides(const int itemsize, const int nd,
                           const npy_intp *dims, const npy_intp *strides,
                           npy_intp *lower_offset, npy_intp *upper_offset);


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
 * return true if pointer is aligned to 'alignment'
 */
static NPY_INLINE int
npy_is_aligned(const void * p, const npy_uintp alignment)
{
    /*
     * alignment is usually a power of two
     * the test is faster than a direct modulo
     */
    if (NPY_LIKELY((alignment & (alignment - 1)) == 0)) {
        return ((npy_uintp)(p) & ((alignment) - 1)) == 0;
    }
    else {
        return ((npy_uintp)(p) % alignment) == 0;
    }
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

static NPY_INLINE int
_is_basic_python_type(PyObject * obj)
{
    if (obj == Py_None ||
            PyBool_Check(obj) ||
            /* Basic number types */
#if !defined(NPY_PY3K)
            PyInt_CheckExact(obj) ||
            PyString_CheckExact(obj) ||
#endif
            PyLong_CheckExact(obj) ||
            PyFloat_CheckExact(obj) ||
            PyComplex_CheckExact(obj) ||
            /* Basic sequence types */
            PyList_CheckExact(obj) ||
            PyTuple_CheckExact(obj) ||
            PyDict_CheckExact(obj) ||
            PyAnySet_CheckExact(obj) ||
            PyUnicode_CheckExact(obj) ||
            PyBytes_CheckExact(obj) ||
            PySlice_Check(obj)) {

        return 1;
    }

    return 0;
}


/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param Dimensionality of the shape
 * @param npy_intp pointer to shape array
 * @param String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
static NPY_INLINE PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUString_FromFormat("()%s", ending);
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        tmp = PyUString_FromFormat(",)%s", ending);
    }
    else {
        tmp = PyUString_FromFormat(")%s", ending);
    }
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}


/*
 * Sets ValueError with "matrices not aligned" message for np.dot and friends
 * when a.shape[i] should match b.shape[j], but doesn't.
 */
static NPY_INLINE void
not_aligned(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUString_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");

    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    shape1_i = PyLong_FromSsize_t(PyArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyArray_DIM(b, j));

    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    errmsg = PyUString_Format(format, fmt_args);
    if (errmsg != NULL) {
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
}

#include "ucsnarrow.h"

#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "common.h"

#include "ctors.h"

#include "shape.h"

#include "buffer.h"

#include "numpymemoryview.h"

#include "lowlevel_strided_loops.h"

/*
 * Reading from a file or a string.
 *
 * As much as possible, we try to use the same code for both files and strings,
 * so the semantics for fromstring and fromfile are the same, especially with
 * regards to the handling of text representations.
 */

typedef int (*next_element)(void **, void *, PyArray_Descr *, void *);
typedef int (*skip_separator)(void **, const char *, void *);

static int
fromstr_next_element(char **s, void *dptr, PyArray_Descr *dtype,
                     const char *end)
{
    int r = dtype->f->fromstr(*s, dptr, s, dtype);
    if (end != NULL && *s > end) {
        return -1;
    }
    return r;
}

static int
fromfile_next_element(FILE **fp, void *dptr, PyArray_Descr *dtype,
                      void *NPY_UNUSED(stream_data))
{
    /* the NULL argument is for backwards-compatibility */
    return dtype->f->scanfunc(*fp, dptr, NULL, dtype);
}

/*
 * Remove multiple whitespace from the separator, and add a space to the
 * beginning and end. This simplifies the separator-skipping code below.
 */
static char *
swab_separator(char *sep)
{
    int skip_space = 0;
    char *s, *start;

    s = start = malloc(strlen(sep)+3);
    /* add space to front if there isn't one */
    if (*sep != '\0' && !isspace(*sep)) {
        *s = ' '; s++;
    }
    while (*sep != '\0') {
        if (isspace(*sep)) {
            if (skip_space) {
                sep++;
            }
            else {
                *s = ' ';
                s++;
                sep++;
                skip_space = 1;
            }
        }
        else {
            *s = *sep;
            s++;
            sep++;
            skip_space = 0;
        }
    }
    /* add space to end if there isn't one */
    if (s != start && s[-1] == ' ') {
        *s = ' ';
        s++;
    }
    *s = '\0';
    return start;
}

/*
 * Assuming that the separator is the next bit in the string (file), skip it.
 *
 * Single spaces in the separator are matched to arbitrary-long sequences
 * of whitespace in the input. If the separator consists only of spaces,
 * it matches one or more whitespace characters.
 *
 * If we can't match the separator, return -2.
 * If we hit the end of the string (file), return -1.
 * Otherwise, return 0.
 */
static int
fromstr_skip_separator(char **s, const char *sep, const char *end)
{
    char *string = *s;
    int result = 0;
    while (1) {
        char c = *string;
        if (c == '\0' || (end != NULL && string >= end)) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            if (string != *s) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                continue;
            }
        }
        else if (*sep != c) {
            result = -2;
            break;
        }
        else {
            sep++;
        }
        string++;
    }
    *s = string;
    return result;
}

static int
fromfile_skip_separator(FILE **fp, const char *sep, void *NPY_UNUSED(stream_data))
{
    int result = 0;
    const char *sep_start = sep;

    while (1) {
        int c = fgetc(*fp);

        if (c == EOF) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            ungetc(c, *fp);
            if (sep != sep_start) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                sep_start++;
                ungetc(c, *fp);
            }
            else if (sep == sep_start) {
                sep_start--;
            }
        }
        else if (*sep != c) {
            ungetc(c, *fp);
            result = -2;
            break;
        }
        else {
            sep++;
        }
    }
    return result;
}

/*
 * Change a sub-array field to the base descriptor
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end.
 *
 * Strides are only added if given (because data is given).
 */
static int
_update_descr_and_dimensions(PyArray_Descr **des, npy_intp *newdims,
                             npy_intp *newstrides, int oldnd)
{
    PyArray_Descr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    int tuple;

    old = *des;
    *des = old->subarray->base;


    mydim = newdims + oldnd;
    tuple = PyTuple_Check(old->subarray->shape);
    if (tuple) {
        numnew = PyTuple_GET_SIZE(old->subarray->shape);
    }
    else {
        numnew = 1;
    }


    newnd = oldnd + numnew;
    if (newnd > MAX_DIMS) {
        goto finish;
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (npy_intp) PyInt_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));
        }
    }
    else {
        mydim[0] = (npy_intp) PyInt_AsLong(old->subarray->shape);
    }

    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;

        mystrides = newstrides + oldnd;
        /* Make new strides -- alwasy C-contiguous */
        tempsize = (*des)->elsize;
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;
            tempsize *= mydim[i] ? mydim[i] : 1;
        }
    }

 finish:
    Py_INCREF(*des);
    Py_DECREF(old);
    return newnd;
}

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;

#define _COPY_N_SIZE(size)                      \
    for(i=0; i<N; i++) {                       \
        memcpy(tout, tin, size);                \
        tin += instrides;                       \
        tout += outstrides;                     \
    }                                           \
    return

    switch(elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE

}

NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size)
{
    char *a, *b, c = 0;
    int j, m;

    switch(size) {
    case 1: /* no byteswap necessary */
        break;
    case 4:
        for (a = (char*)p; n > 0; n--, a += stride - 1) {
            b = a + 3;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 8:
        for (a = (char*)p; n > 0; n--, a += stride - 3) {
            b = a + 7;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 2:
        for (a = (char*)p; n > 0; n--, a += stride) {
            b = a + 1;
            c = *a; *a = *b; *b = c;
        }
        break;
    default:
        m = size/2;
        for (a = (char *)p; n > 0; n--, a += stride - m) {
            b = a + (size - 1);
            for (j = 0; j < m; j++) {
                c=*a; *a++ = *b; *b-- = c;
            }
        }
        break;
    }
}

NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size)
{
    _strided_byte_swap(p, (npy_intp) size, n, size);
    return;
}

/* If numitems > 1, then dst must be contiguous */
NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, npy_intp numitems,
              npy_intp srcstrides, int swap)
{
    npy_intp i;
    char *s1 = (char *)src;
    char *d1 = (char *)dst;


    if ((numitems == 1) || (itemsize == srcstrides)) {
        memcpy(d1, s1, itemsize*numitems);
    }
    else {
        for (i = 0; i < numitems; i++) {
            memcpy(d1, s1, itemsize);
            d1 += itemsize;
            s1 += srcstrides;
        }
    }

    if (swap) {
        byte_swap_vector(d1, numitems, itemsize);
    }
}

/* Gets a half-open range [start, end) which contains the array data */
NPY_NO_EXPORT void
_get_array_memory_extents(PyArrayObject *arr,
                    npy_uintp *out_start, npy_uintp *out_end)
{
    npy_uintp start, end;
    npy_intp idim, ndim = PyArray_NDIM(arr);
    npy_intp *dimensions = PyArray_DIMS(arr),
            *strides = PyArray_STRIDES(arr);

    /* Calculate with a closed range [start, end] */
    start = end = (npy_uintp)PyArray_DATA(arr);
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp stride = strides[idim], dim = dimensions[idim];
        /* If the array size is zero, return an empty range */
        if (dim == 0) {
            *out_start = *out_end = (npy_uintp)PyArray_DATA(arr);
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        else {
            if (stride > 0) {
                end += stride*(dim-1);
            }
            else if (stride < 0) {
                start += stride*(dim-1);
            }
        }
    }

    /* Return a half-open range */
    *out_start = start;
    *out_end = end + arr->descr->elsize;
}

/* Returns 1 if the arrays have overlapping data, 0 otherwise */
NPY_NO_EXPORT int
_arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2)
{
    npy_uintp start1 = 0, start2 = 0, end1 = 0, end2 = 0;

    _get_array_memory_extents(arr1, &start1, &end1);
    _get_array_memory_extents(arr2, &start2, &end2);

    return (start1 < end2) && (start2 < end1);
}

/*NUMPY_API
 * Move the memory of one array into another, allowing for overlapping data.
 *
 * This is in general a difficult problem to solve efficiently, because
 * strides can be negative.  Consider "a = np.arange(3); a[::-1] = a", which
 * previously produced the incorrect [0, 1, 0].
 *
 * Instead of trying to be fancy, we simply check for overlap and make
 * a temporary copy when one exists.
 *
 * Returns 0 on success, negative on failure.
 */
NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dst, PyArrayObject *src)
{
    /*
     * Performance fix for expresions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript_nice in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_ass_sub in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    if (PyArray_DATA(src) == PyArray_DATA(dst) &&
                        PyArray_DESCR(src) == PyArray_DESCR(dst) &&
                        PyArray_NDIM(src) == PyArray_NDIM(dst) &&
                        PyArray_CompareLists(PyArray_DIMS(src),
                                             PyArray_DIMS(dst),
                                             PyArray_NDIM(src)) &&
                        PyArray_CompareLists(PyArray_STRIDES(src),
                                             PyArray_STRIDES(dst),
                                             PyArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        return 0;
    }

    /*
     * A special case is when there is just one dimension with positive
     * strides, and we pass that to CopyInto, which correctly handles
     * it for most cases.  It may still incorrectly handle copying of
     * partially-overlapping data elements, where the data pointer was offset
     * by a fraction of the element size.
     */
    if ((PyArray_NDIM(dst) == 1 &&
                        PyArray_NDIM(src) == 1 &&
                        PyArray_STRIDE(dst, 0) > 0 &&
                        PyArray_STRIDE(src, 0) > 0) ||
                        !_arrays_overlap(dst, src)) {
        return PyArray_CopyInto(dst, src);
    }
    else {
        PyArrayObject *tmp;
        int ret;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray(dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            return -1;
        }
        ret = PyArray_CopyInto(tmp, src);
        if (ret == 0) {
            ret = PyArray_CopyInto(dst, tmp);
        }
        Py_DECREF(tmp);
        return ret;
    }
}



/* adapted from Numarray */
static int
setArrayFromSequence(PyArrayObject *a, PyObject *s, int dim, npy_intp offset)
{
    Py_ssize_t i, slen;
    int res = -1;

    /*
     * This code is to ensure that the sequence access below will
     * return a lower-dimensional sequence.
     */

    /* INCREF on entry DECREF on exit */
    Py_INCREF(s);

    if (PyArray_Check(s) && !(PyArray_CheckExact(s))) {
      /*
       * FIXME:  This could probably copy the entire subarray at once here using
       * a faster algorithm.  Right now, just make sure a base-class array is
       * used so that the dimensionality reduction assumption is correct.
       */
        /* This will DECREF(s) if replaced */
        s = PyArray_EnsureArray(s);
        if (s == NULL) {
            goto fail;
        }
    }

    if (dim > a->nd) {
        PyErr_Format(PyExc_ValueError,
                 "setArrayFromSequence: sequence/array dimensions mismatch.");
        goto fail;
    }

    slen = PySequence_Length(s);
    if (slen < 0) {
        goto fail;
    }
    /*
     * Either the dimensions match, or the sequence has length 1 and can
     * be broadcast to the destination.
     */
    if (slen != a->dimensions[dim] && slen != 1) {
        PyErr_Format(PyExc_ValueError,
                 "cannot copy sequence with size %d to array axis "
                 "with dimension %d", (int)slen, (int)a->dimensions[dim]);
        goto fail;
    }

    /* Broadcast the one element from the sequence to all the outputs */
    if (slen == 1) {
        PyObject *o;
        npy_intp alen = a->dimensions[dim];

        o = PySequence_GetItem(s, 0);
        if (o == NULL) {
            goto fail;
        }
        for (i = 0; i < alen; i++) {
            if ((a->nd - dim) > 1) {
                res = setArrayFromSequence(a, o, dim+1, offset);
            }
            else {
                res = a->descr->f->setitem(o, (a->data + offset), a);
            }
            if (res < 0) {
                Py_DECREF(o);
                goto fail;
            }
            offset += a->strides[dim];
        }
        Py_DECREF(o);
    }
    /* Copy element by element */
    else {
        for (i = 0; i < slen; i++) {
            PyObject *o = PySequence_GetItem(s, i);
            if (o == NULL) {
                goto fail;
            }
            if ((a->nd - dim) > 1) {
                res = setArrayFromSequence(a, o, dim+1, offset);
            }
            else {
                res = a->descr->f->setitem(o, (a->data + offset), a);
            }
            Py_DECREF(o);
            if (res < 0) {
                goto fail;
            }
            offset += a->strides[dim];
        }
    }

    Py_DECREF(s);
    return 0;

 fail:
    Py_DECREF(s);
    return res;
}

NPY_NO_EXPORT int
PyArray_AssignFromSequence(PyArrayObject *self, PyObject *v)
{
    if (!PySequence_Check(v)) {
        PyErr_SetString(PyExc_ValueError,
                        "assignment from non-sequence");
        return -1;
    }
    if (self->nd == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "assignment to 0-d array");
        return -1;
    }
    return setArrayFromSequence(self, v, 0, 0);
}

/*
 * The rest of this code is to build the right kind of array
 * from a python object.
 */

static int
discover_itemsize(PyObject *s, int nd, int *itemsize)
{
    int n, r, i;

    if (PyArray_Check(s)) {
        *itemsize = MAX(*itemsize, PyArray_ITEMSIZE(s));
        return 0;
    }

    if ((nd == 0) || PyString_Check(s) ||
#if defined(NPY_PY3K)
        PyMemoryView_Check(s) ||
#else
        PyBuffer_Check(s) ||
#endif
        PyUnicode_Check(s)) {

        /* If an object has no length, leave it be */
        n = PyObject_Length(s);
        if (n == -1) {
            PyErr_Clear();
        }
        else {
            *itemsize = MAX(*itemsize, n);
        }
        return 0;
    }

    n = PySequence_Length(s);
    for (i = 0; i < n; i++) {
        PyObject *e = PySequence_GetItem(s,i);

        if (e == NULL) {
            return -1;
        }

        r = discover_itemsize(e,nd-1,itemsize);
        Py_DECREF(e);
        if (r == -1) {
            return -1;
        }
    }

    return 0;
}

/*
 * Take an arbitrary object and discover how many dimensions it
 * has, filling in the dimensions as we go.
 */
static int
discover_dimensions(PyObject *s, int *maxndim, npy_intp *d, int check_it,
                                    int stop_at_string, int stop_at_tuple,
                                    int *out_is_object)
{
    PyObject *e;
    int r, n, i;
#if PY_VERSION_HEX >= 0x02060000
    Py_buffer buffer_view;
#endif

    if (*maxndim == 0) {
        return 0;
    }

    /* s is an Array */
    if (PyArray_Check(s)) {
        if (PyArray_NDIM(s) < *maxndim) {
            *maxndim = PyArray_NDIM(s);
        }

        for (i=0; i<*maxndim; i++) {
            d[i] = PyArray_DIM(s,i);
        }
        return 0;
    }

    /* s is a Scalar */
    if (PyArray_IsScalar(s, Generic)) {
        *maxndim = 0;
        return 0;
    }

    /* s is not a Sequence */
    if (!PySequence_Check(s) ||
#if defined(NPY_PY3K)
        /* FIXME: XXX -- what is the correct thing to do here? */
#else
            PyInstance_Check(s) ||
#endif
            PySequence_Length(s) < 0) {
        *maxndim = 0;
        PyErr_Clear();
        return 0;
    }

    /* s is a String */
    if (PyString_Check(s) ||
#if defined(NPY_PY3K)
#else
            PyBuffer_Check(s) ||
#endif
            PyUnicode_Check(s)) {
        if (stop_at_string) {
            *maxndim = 0;
        }
        else {
            d[0] = PySequence_Length(s);
            *maxndim = 1;
        }
        return 0;
    }

    /* s is a Tuple, but tuples aren't expanded */
    if (stop_at_tuple && PyTuple_Check(s)) {
        *maxndim = 0;
        return 0;
    }

    /* s is a PEP 3118 buffer */
#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 buffer interface */
    memset(&buffer_view, 0, sizeof(Py_buffer));
    if (PyObject_GetBuffer(s, &buffer_view, PyBUF_STRIDES) == 0 ||
        PyObject_GetBuffer(s, &buffer_view, PyBUF_ND) == 0) {
        int nd = buffer_view.ndim;
        if (nd < *maxndim) {
            *maxndim = nd;
        }
        for (i=0; i<*maxndim; i++) {
            d[i] = buffer_view.shape[i];
        }
        PyBuffer_Release(&buffer_view);
        return 0;
    }
    else if (PyObject_GetBuffer(s, &buffer_view, PyBUF_SIMPLE) == 0) {
        d[0] = buffer_view.len;
        *maxndim = 1;
        PyBuffer_Release(&buffer_view);
        return 0;
    }
    else {
        PyErr_Clear();
    }
#endif

    /* s has the __array_struct__ interface */
    if ((e = PyObject_GetAttrString(s, "__array_struct__")) != NULL) {
        int nd = -1;
        if (NpyCapsule_Check(e)) {
            PyArrayInterface *inter;
            inter = (PyArrayInterface *)NpyCapsule_AsVoidPtr(e);
            if (inter->two == 2) {
                nd = inter->nd;
                if (nd >= 0) {
                    if (nd < *maxndim) {
                        *maxndim = nd;
                    }
                    for (i=0; i<*maxndim; i++) {
                        d[i] = inter->shape[i];
                    }
                }
            }
        }
        Py_DECREF(e);
        if (nd >= 0) {
            return 0;
        }
    }
    else {
        PyErr_Clear();
    }

    /* s has the __array_interface__ interface */
    if ((e = PyObject_GetAttrString(s, "__array_interface__")) != NULL) {
        int nd = -1;
        if (PyDict_Check(e)) {
            PyObject *new;
            new = PyDict_GetItemString(e, "shape");
            if (new && PyTuple_Check(new)) {
                nd = PyTuple_GET_SIZE(new);
                if (nd < *maxndim) {
                    *maxndim = nd;
                }
                for (i=0; i<*maxndim; i++) {
#if (PY_VERSION_HEX >= 0x02050000)
                    d[i] = PyInt_AsSsize_t(PyTuple_GET_ITEM(new, i));
#else
                    d[i] = PyInt_AsLong(PyTuple_GET_ITEM(new, i));
#endif
                    if (d[i] < 0) {
                        PyErr_SetString(PyExc_RuntimeError,
                                "Invalid shape in __array_interface__");
                        Py_DECREF(e);
                        return -1;
                    }
                }
            }
        }
        Py_DECREF(e);
        if (nd >= 0) {
            return 0;
        }
    }
    else {
        PyErr_Clear();
    }

    n = PySequence_Size(s);

    if (n < 0) {
        return -1;
    }

    d[0] = n;

    /* 1-dimensional sequence */
    if (n == 0 || *maxndim == 1) {
        *maxndim = 1;
        return 0;
    }
    else {
        npy_intp dtmp[NPY_MAXDIMS];
        int j, maxndim_m1 = *maxndim - 1;

        if ((e = PySequence_GetItem(s, 0)) == NULL) {
            /*
             * PySequence_Check detects whether an old type object is a
             * sequence by the presence of the __getitem__ attribute, and
             * for new type objects that aren't dictionaries by the
             * presence of the __len__ attribute as well. In either case it
             * is possible to have an object that tests as a sequence but
             * doesn't behave as a sequence and consequently, the
             * PySequence_GetItem call can fail. When that happens and the
             * object looks like a dictionary, we truncate the dimensions
             * and set the object creation flag, otherwise we pass the
             * error back up the call chain.
             */
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                PyErr_Clear();
                *maxndim = 0;
                *out_is_object = 1;
                return 0;
            }
            else {
                return -1;
            }
        }
        r = discover_dimensions(e, &maxndim_m1, d + 1, check_it,
                                        stop_at_string, stop_at_tuple,
                                        out_is_object);
        Py_DECREF(e);
        if (r < 0) {
            return r;
        }

        /* For the dimension truncation check below */
        *maxndim = maxndim_m1 + 1;
        for (i = 1; i < n; ++i) {
            /* Get the dimensions of the first item */
            if ((e = PySequence_GetItem(s, i)) == NULL) {
                /* see comment above */
                if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                    PyErr_Clear();
                    *maxndim = 0;
                    *out_is_object = 1;
                    return 0;
                }
                else {
                    return -1;
                }
            }
            r = discover_dimensions(e, &maxndim_m1, dtmp, check_it,
                                            stop_at_string, stop_at_tuple,
                                            out_is_object);
            Py_DECREF(e);
            if (r < 0) {
                return r;
            }

            /* Reduce max_ndim_m1 to just items which match */
            for (j = 0; j < maxndim_m1; ++j) {
                if (dtmp[j] != d[j+1]) {
                    maxndim_m1 = j;
                    break;
                }
            }
        }
        /*
         * If the dimensions are truncated, need to produce
         * an object array.
         */
        if (maxndim_m1 + 1 < *maxndim) {
            *out_is_object = 1;
            *maxndim = maxndim_m1 + 1;
        }
    }

    return 0;
}

/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr (even on failure)
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     npy_intp *dims, npy_intp *strides, void *data,
                     int flags, PyObject *obj)
{
    PyArrayObject *self;
    int i;
    size_t sd;
    npy_intp largest;
    npy_intp size;

    if (descr->subarray) {
        PyObject *ret;
        npy_intp newdims[2*NPY_MAXDIMS];
        npy_intp *newstrides = NULL;
        memcpy(newdims, dims, nd*sizeof(npy_intp));
        if (strides) {
            newstrides = newdims + NPY_MAXDIMS;
            memcpy(newstrides, strides, nd*sizeof(npy_intp));
        }
        nd =_update_descr_and_dimensions(&descr, newdims,
                                         newstrides, nd);
        ret = PyArray_NewFromDescr(subtype, descr, nd, newdims,
                                   newstrides,
                                   data, flags, obj);
        return ret;
    }

    if ((unsigned int)nd > (unsigned int)NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                     "number of dimensions must be within [0, %d]",
                     NPY_MAXDIMS);
        Py_DECREF(descr);
        return NULL;
    }

    /* Check dimensions */
    size = 1;
    sd = (size_t) descr->elsize;
    if (sd == 0) {
        if (!PyDataType_ISSTRING(descr)) {
            PyErr_SetString(PyExc_TypeError, "Empty data-type");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        if (descr->type_num == NPY_STRING) {
            sd = descr->elsize = 1;
        }
        else {
            sd = descr->elsize = sizeof(PyArray_UCS4);
        }
    }

    largest = NPY_MAX_INTP / sd;
    for (i = 0; i < nd; i++) {
        npy_intp dim = dims[i];

        if (dim == 0) {
            /*
             * Compare to PyArray_OverflowMultiplyList that
             * returns 0 in this case.
             */
            continue;
        }

        if (dim < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "negative dimensions "  \
                            "are not allowed");
            Py_DECREF(descr);
            return NULL;
        }

        if (dim > largest) {
            PyErr_SetString(PyExc_ValueError,
                            "array is too big.");
            Py_DECREF(descr);
            return NULL;
        }
        size *= dim;
        largest /= dim;
    }

    self = (PyArrayObject *) subtype->tp_alloc(subtype, 0);
    if (self == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    self->nd = nd;
    self->dimensions = NULL;
    self->data = NULL;
    if (data == NULL) {
        self->flags = DEFAULT;
        if (flags) {
            self->flags |= NPY_F_CONTIGUOUS;
            if (nd > 1) {
                self->flags &= ~NPY_C_CONTIGUOUS;
            }
            flags = NPY_F_CONTIGUOUS;
        }
    }
    else {
        self->flags = (flags & ~NPY_UPDATEIFCOPY);
    }
    self->descr = descr;
    self->base = (PyObject *)NULL;
    self->weakreflist = (PyObject *)NULL;

    if (nd > 0) {
        self->dimensions = PyDimMem_NEW(2*nd);
        if (self->dimensions == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        self->strides = self->dimensions + nd;
        memcpy(self->dimensions, dims, sizeof(npy_intp)*nd);
        if (strides == NULL) { /* fill it in */
            sd = _array_fill_strides(self->strides, dims, nd, sd,
                                     flags, &(self->flags));
        }
        else {
            /*
             * we allow strides even when we create
             * the memory, but be careful with this...
             */
            memcpy(self->strides, strides, sizeof(npy_intp)*nd);
            sd *= size;
        }
    }
    else {
        self->dimensions = self->strides = NULL;
        self->flags |= NPY_F_CONTIGUOUS;
    }

    if (data == NULL) {
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         */

        if (sd == 0) {
            sd = descr->elsize;
        }
        data = PyDataMem_NEW(sd);
        if (data == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        self->flags |= OWNDATA;

        /*
         * It is bad to have unitialized OBJECT pointers
         * which could also be sub-fields of a VOID array
         */
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(data, 0, sd);
        }
    }
    else {
        /*
         * If data is passed in, this object won't own it by default.
         * Caller must arrange for this to be reset if truly desired
         */
        self->flags &= ~OWNDATA;
    }
    self->data = data;

    /*
     * If the strides were provided to the function, need to
     * update the flags to get the right CONTIGUOUS, ALIGN properties
     */
    if (strides != NULL) {
        PyArray_UpdateFlags(self, UPDATE_ALL);
    }

    /*
     * call the __array_finalize__
     * method if a subtype.
     * If obj is NULL, then call method with Py_None
     */
    if ((subtype != &PyArray_Type)) {
        PyObject *res, *func, *args;

        func = PyObject_GetAttrString((PyObject *)self, "__array_finalize__");
        if (func && func != Py_None) {
            if (NpyCapsule_Check(func)) {
                /* A C-function is stored here */
                PyArray_FinalizeFunc *cfunc;
                cfunc = NpyCapsule_AsVoidPtr(func);
                Py_DECREF(func);
                if (cfunc(self, obj) < 0) {
                    goto fail;
                }
            }
            else {
                args = PyTuple_New(1);
                if (obj == NULL) {
                    obj=Py_None;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(args, 0, obj);
                res = PyObject_Call(func, args, NULL);
                Py_DECREF(args);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
        else Py_XDECREF(func);
    }
    return (PyObject *)self;

 fail:
    Py_DECREF(self);
    return NULL;
}

/*NUMPY_API
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order and data type changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * dtype     - If not NULL, overrides the data type of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArray(PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok)
{
    PyObject *ret = NULL;
    int ndim = PyArray_NDIM(prototype);

    /* If no override data type, use the one from the prototype */
    if (dtype == NULL) {
        dtype = PyArray_DESCR(prototype);
        Py_INCREF(dtype);
    }

    /* Handle ANYORDER and simple KEEPORDER cases */
    switch (order) {
        case NPY_ANYORDER:
            order = PyArray_ISFORTRAN(prototype) ?
                                    NPY_FORTRANORDER : NPY_CORDER;
            break;
        case NPY_KEEPORDER:
            if (PyArray_IS_C_CONTIGUOUS(prototype) || ndim <= 1) {
                order = NPY_CORDER;
                break;
            }
            else if (PyArray_IS_F_CONTIGUOUS(prototype)) {
                order = NPY_FORTRANORDER;
                break;
            }
            break;
        default:
            break;
    }

    /* If it's not KEEPORDER, this is simple */
    if (order != NPY_KEEPORDER) {
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        dtype,
                                        ndim,
                                        PyArray_DIMS(prototype),
                                        NULL,
                                        NULL,
                                        order,
                                        subok ? (PyObject *)prototype : NULL);
    }
    /* KEEPORDER needs some analysis of the strides */
    else {
        npy_intp strides[NPY_MAXDIMS], stride;
        npy_intp *shape = PyArray_DIMS(prototype);
        _npy_stride_sort_item strideperm[NPY_MAXDIMS];
        int i;

        PyArray_CreateSortedStridePerm(prototype, strideperm);

        /* Build the new strides */
        stride = dtype->elsize;
        for (i = ndim-1; i >= 0; --i) {
            npy_intp i_perm = strideperm[i].perm;
            strides[i_perm] = stride;
            stride *= shape[i_perm];
        }

        /* Finally, allocate the array */
        ret = PyArray_NewFromDescr( subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        dtype,
                                        ndim,
                                        shape,
                                        strides,
                                        NULL,
                                        0,
                                        subok ? (PyObject *)prototype : NULL);
    }

    return ret;
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(PyTypeObject *subtype, int nd, npy_intp *dims, int type_num,
            npy_intp *strides, void *data, int itemsize, int flags,
            PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (descr->elsize == 0) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        descr->elsize = itemsize;
    }
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}


NPY_NO_EXPORT int
_array_from_buffer_3118(PyObject *obj, PyObject **out)
{
#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 */
    PyObject *memoryview;
    Py_buffer *view;
    PyArray_Descr *descr = NULL;
    PyObject *r;
    int nd, flags, k;
    Py_ssize_t d;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    memoryview = PyMemoryView_FromObject(obj);
    if (memoryview == NULL) {
        PyErr_Clear();
        return -1;
    }

    view = PyMemoryView_GET_BUFFER(memoryview);
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            PyObject *msg;
            msg = PyBytes_FromFormat("Invalid PEP 3118 format string: '%s'",
                                     view->format);
            PyErr_WarnEx(PyExc_RuntimeWarning, PyBytes_AS_STRING(msg), 0);
            Py_DECREF(msg);
            goto fail;
        }

        /* Sanity check */
        if (descr->elsize != view->itemsize) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Item size computed from the PEP 3118 buffer format "
                         "string does not match the actual item size.",
                         0);
            goto fail;
        }
    }
    else {
        descr = PyArray_DescrNewFromType(PyArray_STRING);
        descr->elsize = view->itemsize;
    }

    if (view->shape != NULL) {
        nd = view->ndim;
        if (nd >= NPY_MAXDIMS || nd < 0) {
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
            if (k >= NPY_MAXDIMS) {
                goto fail;
            }
            shape[k] = view->shape[k];
        }
        if (view->strides != NULL) {
            for (k = 0; k < nd; ++k) {
                strides[k] = view->strides[k];
            }
        }
        else {
            d = view->len;
            for (k = 0; k < nd; ++k) {
                d /= view->shape[k];
                strides[k] = d;
            }
        }
    }
    else {
        nd = 1;
        shape[0] = view->len / view->itemsize;
        strides[0] = view->itemsize;
    }

    flags = BEHAVED & (view->readonly ? ~NPY_WRITEABLE : ~0);
    r = PyArray_NewFromDescr(&PyArray_Type, descr,
                             nd, shape, strides, view->buf,
                             flags, NULL);
    ((PyArrayObject *)r)->base = memoryview;
    PyArray_UpdateFlags((PyArrayObject *)r, UPDATE_ALL);

    *out = r;
    return 0;

fail:
    Py_XDECREF(descr);
    Py_DECREF(memoryview);
    return -1;

#else
    return -1;
#endif
}

/*NUMPY_API
 * Retrieves the array parameters for viewing/converting an arbitrary
 * PyObject* to a NumPy array. This allows the "innate type and shape"
 * of Python list-of-lists to be discovered without
 * actually converting to an array.
 *
 * In some cases, such as structured arrays and the __array__ interface,
 * a data type needs to be used to make sense of the object.  When
 * this is needed, provide a Descr for 'requested_dtype', otherwise
 * provide NULL. This reference is not stolen. Also, if the requested
 * dtype doesn't modify the interpretation of the input, out_dtype will
 * still get the "innate" dtype of the object, not the dtype passed
 * in 'requested_dtype'.
 *
 * If writing to the value in 'op' is desired, set the boolean
 * 'writeable' to 1.  This raises an error when 'op' is a scalar, list
 * of lists, or other non-writeable 'op'.
 *
 * Result: When success (0 return value) is returned, either out_arr
 *         is filled with a non-NULL PyArrayObject and
 *         the rest of the parameters are untouched, or out_arr is
 *         filled with NULL, and the rest of the parameters are
 *         filled.
 *
 * Typical usage:
 *
 *      PyArrayObject *arr = NULL;
 *      PyArray_Descr *dtype = NULL;
 *      int ndim = 0;
 *      npy_intp dims[NPY_MAXDIMS];
 *
 *      if (PyArray_GetArrayParamsFromObject(op, NULL, 1, &dtype,
 *                                          &ndim, &dims, &arr, NULL) < 0) {
 *          return NULL;
 *      }
 *      if (arr == NULL) {
 *          ... validate/change dtype, validate flags, ndim, etc ...
 *          // Could make custom strides here too
 *          arr = PyArray_NewFromDescr(&PyArray_Type, dtype, ndim,
 *                                      dims, NULL,
 *                                      fortran ? NPY_F_CONTIGUOUS : 0,
 *                                      NULL);
 *          if (arr == NULL) {
 *              return NULL;
 *          }
 *          if (PyArray_CopyObject(arr, op) < 0) {
 *              Py_DECREF(arr);
 *              return NULL;
 *          }
 *      }
 *      else {
 *          ... in this case the other parameters weren't filled, just
 *              validate and possibly copy arr itself ...
 *      }
 *      ... use arr ...
 */
NPY_NO_EXPORT int
PyArray_GetArrayParamsFromObject(PyObject *op,
                        PyArray_Descr *requested_dtype,
                        npy_bool writeable,
                        PyArray_Descr **out_dtype,
                        int *out_ndim, npy_intp *out_dims,
                        PyArrayObject **out_arr, PyObject *context)
{
    PyObject *tmp;

    /* If op is an array */
    if (PyArray_Check(op)) {
        if (writeable && !PyArray_ISWRITEABLE((PyArrayObject *)op)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "cannot write to array");
            return -1;
        }
        Py_INCREF(op);
        *out_arr = (PyArrayObject *)op;
        return 0;
    }

    /* If op is a NumPy scalar */
    if (PyArray_IsScalar(op, Generic)) {
        if (writeable) {
            PyErr_SetString(PyExc_RuntimeError,
                                "cannot write to scalar");
            return -1;
        }
        *out_dtype = PyArray_DescrFromScalar(op);
        if (*out_dtype == NULL) {
            return -1;
        }
        *out_ndim = 0;
        *out_arr = NULL;
        return 0;
    }

    /* If op is a Python scalar */
    *out_dtype = _array_find_python_scalar_type(op);
    if (*out_dtype != NULL) {
        if (writeable) {
            PyErr_SetString(PyExc_RuntimeError,
                                "cannot write to scalar");
            Py_DECREF(*out_dtype);
            return -1;
        }
        *out_ndim = 0;
        *out_arr = NULL;
        return 0;
    }

    /* If op supports the PEP 3118 buffer interface */
    if (!PyBytes_Check(op) && !PyUnicode_Check(op) &&
             _array_from_buffer_3118(op, (PyObject **)out_arr) == 0) {
        if (writeable && !PyArray_ISWRITEABLE(*out_arr)) {
            PyErr_SetString(PyExc_RuntimeError,
                                "cannot write to PEP 3118 buffer");
            Py_DECREF(*out_arr);
            return -1;
        }
        return (*out_arr) == NULL ? -1 : 0;
    }

    /* If op supports the __array_struct__ or __array_interface__ interface */
    tmp = PyArray_FromStructInterface(op);
    if (tmp == Py_NotImplemented) {
        tmp = PyArray_FromInterface(op);
    }
    if (tmp != Py_NotImplemented) {
        if (writeable && !PyArray_ISWRITEABLE(tmp)) {
            PyErr_SetString(PyExc_RuntimeError,
                                "cannot write to array interface object");
            Py_DECREF(tmp);
            return -1;
        }
        *out_arr = (PyArrayObject *)tmp;
        return (*out_arr) == NULL ? -1 : 0;
    }

    /*
     * If op supplies the __array__ function.
     * The documentation says this should produce a copy, so
     * we skip this method if writeable is true, because the intent
     * of writeable is to modify the operand.
     * XXX: If the implementation is wrong, and/or if actual
     *      usage requires this behave differently,
     *      this should be changed!
     */
    if (!writeable) {
        tmp = PyArray_FromArrayAttr(op, requested_dtype, context);
        if (tmp != Py_NotImplemented) {
            if (writeable && !PyArray_ISWRITEABLE(tmp)) {
                PyErr_SetString(PyExc_RuntimeError,
                                    "cannot write to array interface object");
                Py_DECREF(tmp);
                return -1;
            }
            *out_arr = (PyArrayObject *)tmp;
            return (*out_arr) == NULL ? -1 : 0;
        }
    }

    /* Try to treat op as a list of lists */
    if (!writeable && PySequence_Check(op)) {
        int check_it, stop_at_string, stop_at_tuple, is_object;
        int type_num, type;

        /*
         * Determine the type, using the requested data type if
         * it will affect how the array is retrieved
         */
        if (requested_dtype != NULL && (
                requested_dtype->type_num == NPY_STRING ||
                requested_dtype->type_num == NPY_UNICODE ||
                (requested_dtype->type_num == NPY_VOID &&
                    (requested_dtype->names || requested_dtype->subarray)) ||
                requested_dtype->type == NPY_CHARLTR ||
                requested_dtype->type_num == NPY_OBJECT)) {
            Py_INCREF(requested_dtype);
            *out_dtype = requested_dtype;
        }
        else {
            *out_dtype = _array_find_type(op, NULL, MAX_DIMS);
            if (*out_dtype == NULL) {
                if (PyErr_Occurred() &&
                        PyErr_GivenExceptionMatches(PyErr_Occurred(),
                                                PyExc_MemoryError)) {
                    return -1;
                }
                /* Say it's an OBJECT array if there's an error */
                PyErr_Clear();
                *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
                if (*out_dtype == NULL) {
                    return -1;
                }
            }
        }

        type_num = (*out_dtype)->type_num;
        type = (*out_dtype)->type;

        check_it = (type != NPY_CHARLTR);
        stop_at_string = (type_num != NPY_STRING) ||
                         (type == NPY_STRINGLTR);
        stop_at_tuple = (type_num == NPY_VOID &&
                         ((*out_dtype)->names || (*out_dtype)->subarray));

        *out_ndim = NPY_MAXDIMS;
        is_object = 0;
        if (discover_dimensions(op, out_ndim, out_dims, check_it,
                                    stop_at_string, stop_at_tuple,
                                    &is_object) < 0) {
            Py_DECREF(*out_dtype);
            if (PyErr_Occurred()) {
                return -1;
            }
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
            *out_ndim = 0;
            *out_arr = NULL;
            return 0;
        }
        /* If object arrays are forced */
        if (is_object) {
            Py_DECREF(*out_dtype);
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
        }

        if ((*out_dtype)->type == NPY_CHARLTR && (*out_ndim) > 0 &&
                                        out_dims[(*out_ndim) - 1] == 1) {
            (*out_ndim) -= 1;
        }

        /* If the type is flexible, determine its size */
        if ((*out_dtype)->elsize == 0 &&
                            PyTypeNum_ISEXTENDED((*out_dtype)->type_num)) {
            int itemsize = 0;
            if (discover_itemsize(op, *out_ndim, &itemsize) < 0) {
                Py_DECREF(*out_dtype);
                if (PyErr_Occurred() &&
                        PyErr_GivenExceptionMatches(PyErr_Occurred(),
                                                PyExc_MemoryError)) {
                    return -1;
                }
                /* Say it's an OBJECT scalar if there's an error */
                PyErr_Clear();
                *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
                *out_ndim = 0;
                *out_arr = NULL;
                return 0;
            }
            if ((*out_dtype)->type_num == NPY_UNICODE) {
                itemsize *= 4;
            }

            if (itemsize != (*out_dtype)->elsize) {
                PyArray_DESCR_REPLACE(*out_dtype);
                (*out_dtype)->elsize = itemsize;
            }
        }

        *out_arr = NULL;
        return 0;
    }

    /* Anything can be viewed as an object, unless it needs to be writeable */
    if (!writeable) {
        *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
        if (*out_dtype == NULL) {
            return -1;
        }
        *out_ndim = 0;
        *out_arr = NULL;
        return 0;
    }

    PyErr_SetString(PyExc_RuntimeError,
                    "object cannot be viewed as a writeable numpy array");
    return -1;
}

/*NUMPY_API
 * Does not check for NPY_ENSURECOPY and NPY_NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from many different places.
     */
    PyArrayObject *arr = NULL, *ret;
    PyArray_Descr *dtype = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    /* Get either the array or its parameters if it isn't an array */
    if (PyArray_GetArrayParamsFromObject(op, newtype,
                        0, &dtype,
                        &ndim, dims, &arr, context) < 0) {
        Py_XDECREF(newtype);
        ret = NULL;
        return NULL;
    }

    /* If the requested dtype is flexible, adjust its size */
    if (newtype != NULL && newtype->elsize == 0) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            ret = NULL;
            return NULL;
        }
        if (arr != NULL) {
            dtype = PyArray_DESCR(arr);
        }

        if (newtype->type_num == dtype->type_num) {
            newtype->elsize = dtype->elsize;
        }
        else {
            switch(newtype->type_num) {
                case NPY_STRING:
                    if (dtype->type_num == NPY_UNICODE) {
                        newtype->elsize = dtype->elsize >> 2;
                    }
                    else {
                        newtype->elsize = dtype->elsize;
                    }
                    break;
                case NPY_UNICODE:
                    newtype->elsize = dtype->elsize << 2;
                    break;
                case NPY_VOID:
                    newtype->elsize = dtype->elsize;
                    break;
            }
        }
    }

    /* If we got dimensions and dtype instead of an array */
    if (arr == NULL) {
        if (flags&NPY_UPDATEIFCOPY) {
            Py_XDECREF(newtype);
            PyErr_SetString(PyExc_TypeError,
                            "UPDATEIFCOPY used for non-array input.");
            return NULL;
        }
        else if (min_depth != 0 && ndim < min_depth) {
            Py_DECREF(dtype);
            Py_XDECREF(newtype);
            PyErr_SetString(PyExc_ValueError,
                            "object of too small depth for desired array");
            ret = NULL;
        }
        else if (max_depth != 0 && ndim > max_depth) {
            Py_DECREF(dtype);
            Py_XDECREF(newtype);
            PyErr_SetString(PyExc_ValueError,
                            "object too deep for desired array");
            ret = NULL;
        }
        else if (ndim == 0 && PyArray_IsScalar(op, Generic)) {
            ret = (PyArrayObject *)PyArray_FromScalar(op, newtype);
            Py_DECREF(dtype);
        }
        else {
            if (newtype == NULL) {
                newtype = dtype;
            }
            else {
                /*
                 * TODO: would be nice to do this too, but it's
                 *       a behavior change.  It's also a bit tricky
                 *       for downcasting to small integer and float
                 *       types, and might be better to modify
                 *       PyArray_AssignFromSequence and descr->f->setitem
                 *       to have a 'casting' parameter and
                 *       to check each value with scalar rules like
                 *       in PyArray_MinScalarType.
                 */
                /*
                if (!(flags&NPY_FORCECAST) && ndim > 0 &&
                        !PyArray_CanCastTo(dtype, newtype)) {
                    Py_DECREF(dtype);
                    Py_XDECREF(newtype);
                    PyErr_SetString(PyExc_TypeError,
                                "object cannot be safely cast to array "
                                "of required type");
                    return NULL;
                }
                */
                Py_DECREF(dtype);
            }

            /* Create an array and copy the data */
            ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, newtype,
                                                 ndim, dims,
                                                 NULL, NULL,
                                                 flags&NPY_F_CONTIGUOUS, NULL);
            if (ret != NULL) {
                if (ndim > 0) {
                    if (PyArray_AssignFromSequence(ret, op) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                    }
                }
                else {
                    if (PyArray_DESCR(ret)->f->setitem(op,
                                                PyArray_DATA(ret), ret) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                    }
                }
            }
        }
    }
    else {
        if (min_depth != 0 && PyArray_NDIM(arr) < min_depth) {
            PyErr_SetString(PyExc_ValueError,
                            "object of too small depth for desired array");
            Py_DECREF(arr);
            ret = NULL;
        }
        else if (max_depth != 0 && PyArray_NDIM(arr) > max_depth) {
            PyErr_SetString(PyExc_ValueError,
                            "object too deep for desired array");
            Py_DECREF(arr);
            ret = NULL;
        }
        else {
            ret = (PyArrayObject *)PyArray_FromArray(arr, newtype, flags);
            Py_DECREF(arr);
        }
    }

    return (PyObject *)ret;
}

/*
 * flags is any of
 * NPY_C_CONTIGUOUS (CONTIGUOUS),
 * NPY_F_CONTIGUOUS (FORTRAN),
 * NPY_ALIGNED,
 * NPY_WRITEABLE,
 * NPY_NOTSWAPPED,
 * NPY_ENSURECOPY,
 * NPY_UPDATEIFCOPY,
 * NPY_FORCECAST,
 * NPY_ENSUREARRAY,
 * NPY_ELEMENTSTRIDES
 *
 * or'd (|) together
 *
 * Any of these flags present means that the returned array should
 * guarantee that aspect of the array.  Otherwise the returned array
 * won't guarantee it -- it will depend on the object as to whether or
 * not it has such features.
 *
 * Note that NPY_ENSURECOPY is enough
 * to guarantee NPY_C_CONTIGUOUS, NPY_ALIGNED and NPY_WRITEABLE
 * and therefore it is redundant to include those as well.
 *
 * NPY_BEHAVED == NPY_ALIGNED | NPY_WRITEABLE
 * NPY_CARRAY = NPY_C_CONTIGUOUS | NPY_BEHAVED
 * NPY_FARRAY = NPY_F_CONTIGUOUS | NPY_BEHAVED
 *
 * NPY_F_CONTIGUOUS can be set in the FLAGS to request a FORTRAN array.
 * Fortran arrays are always behaved (aligned,
 * notswapped, and writeable) and not (C) CONTIGUOUS (if > 1d).
 *
 * NPY_UPDATEIFCOPY flag sets this flag in the returned array if a copy is
 * made and the base argument points to the (possibly) misbehaved array.
 * When the new array is deallocated, the original array held in base
 * is updated with the contents of the new array.
 *
 * NPY_FORCECAST will cause a cast to occur regardless of whether or not
 * it is safe.
 */

/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    PyObject *obj;
    if (requires & NPY_NOTSWAPPED) {
        if (!descr && PyArray_Check(op) &&
            !PyArray_ISNBO(PyArray_DESCR(op)->byteorder)) {
            descr = PyArray_DescrNew(PyArray_DESCR(op));
        }
        else if (descr && !PyArray_ISNBO(descr->byteorder)) {
            PyArray_DESCR_REPLACE(descr);
        }
        if (descr) {
            descr->byteorder = PyArray_NATIVE;
        }
    }

    obj = PyArray_FromAny(op, descr, min_depth, max_depth, requires, context);
    if (obj == NULL) {
        return NULL;
    }
    if ((requires & NPY_ELEMENTSTRIDES) &&
        !PyArray_ElementStrides(obj)) {
        PyObject *new;
        new = PyArray_NewCopy((PyArrayObject *)obj, NPY_ANYORDER);
        Py_DECREF(obj);
        obj = new;
    }
    return obj;
}

/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags)
{

    PyArrayObject *ret = NULL;
    int itemsize;
    int copy = 0;
    int arrflags;
    PyArray_Descr *oldtype;
    char *msg = "cannot copy back to a read-only array";
    PyTypeObject *subtype;

    oldtype = PyArray_DESCR(arr);
    subtype = Py_TYPE(arr);
    if (newtype == NULL) {
        newtype = oldtype; Py_INCREF(oldtype);
    }
    itemsize = newtype->elsize;
    if (itemsize == 0) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
        itemsize = newtype->elsize;
    }

    /*
     * Can't cast unless ndim-0 array, NPY_FORCECAST is specified
     * or the cast is safe.
     */
    if (!(flags & NPY_FORCECAST) && !PyArray_NDIM(arr) == 0 &&
        !PyArray_CanCastTo(oldtype, newtype)) {
        Py_DECREF(newtype);
        PyErr_SetString(PyExc_TypeError,
                        "array cannot be safely cast "  \
                        "to required type");
        return NULL;
    }

    /* Don't copy if sizes are compatible */
    if ((flags & NPY_ENSURECOPY) || PyArray_EquivTypes(oldtype, newtype)) {
        arrflags = arr->flags;
        if (arr->nd <= 1 && (flags & NPY_F_CONTIGUOUS)) {
            flags |= NPY_C_CONTIGUOUS;
        }
        copy = (flags & NPY_ENSURECOPY) ||
            ((flags & NPY_C_CONTIGUOUS) && (!(arrflags & NPY_C_CONTIGUOUS)))
            || ((flags & NPY_ALIGNED) && (!(arrflags & NPY_ALIGNED)))
            || (arr->nd > 1 &&
                ((flags & NPY_F_CONTIGUOUS) &&
                 (!(arrflags & NPY_F_CONTIGUOUS))))
            || ((flags & NPY_WRITEABLE) && (!(arrflags & NPY_WRITEABLE)));

        if (copy) {
            if ((flags & NPY_UPDATEIFCOPY) &&
                (!PyArray_ISWRITEABLE(arr))) {
                Py_DECREF(newtype);
                PyErr_SetString(PyExc_ValueError, msg);
                return NULL;
            }
            if ((flags & NPY_ENSUREARRAY)) {
                subtype = &PyArray_Type;
            }
            ret = (PyArrayObject *)
                PyArray_NewFromDescr(subtype, newtype,
                                     arr->nd,
                                     arr->dimensions,
                                     NULL, NULL,
                                     flags & NPY_F_CONTIGUOUS,
                                     (PyObject *)arr);
            if (ret == NULL) {
                return NULL;
            }
            if (PyArray_CopyInto(ret, arr) == -1) {
                Py_DECREF(ret);
                return NULL;
            }
            if (flags & NPY_UPDATEIFCOPY)  {
                ret->flags |= NPY_UPDATEIFCOPY;
                ret->base = (PyObject *)arr;
                PyArray_FLAGS(ret->base) &= ~NPY_WRITEABLE;
                Py_INCREF(arr);
            }
        }
        /*
         * If no copy then just increase the reference
         * count and return the input
         */
        else {
            Py_DECREF(newtype);
            if ((flags & NPY_ENSUREARRAY) &&
                !PyArray_CheckExact(arr)) {
                Py_INCREF(arr->descr);
                ret = (PyArrayObject *)
                    PyArray_NewFromDescr(&PyArray_Type,
                                         arr->descr,
                                         arr->nd,
                                         arr->dimensions,
                                         arr->strides,
                                         arr->data,
                                         arr->flags,NULL);
                if (ret == NULL) {
                    return NULL;
                }
                ret->base = (PyObject *)arr;
            }
            else {
                ret = arr;
            }
            Py_INCREF(arr);
        }
    }

    /*
     * The desired output type is different than the input
     * array type and copy was not specified
     */
    else {
        if ((flags & NPY_UPDATEIFCOPY) &&
            (!PyArray_ISWRITEABLE(arr))) {
            Py_DECREF(newtype);
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        if ((flags & NPY_ENSUREARRAY)) {
            subtype = &PyArray_Type;
        }
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(subtype, newtype,
                                 arr->nd, arr->dimensions,
                                 NULL, NULL,
                                 flags & NPY_F_CONTIGUOUS,
                                 (PyObject *)arr);
        if (ret == NULL) {
            return NULL;
        }
        if (PyArray_CastTo(ret, arr) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        if (flags & NPY_UPDATEIFCOPY)  {
            ret->flags |= NPY_UPDATEIFCOPY;
            ret->base = (PyObject *)arr;
            PyArray_FLAGS(ret->base) &= ~NPY_WRITEABLE;
            Py_INCREF(arr);
        }
    }
    return (PyObject *)ret;
}

/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    PyArray_Descr *thetype = NULL;
    char buf[40];
    PyArrayInterface *inter;
    PyObject *attr, *r;
    char endian = PyArray_NATBYTE;

    attr = PyObject_GetAttrString(input, "__array_struct__");
    if (attr == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!NpyCapsule_Check(attr)) {
        goto fail;
    }
    inter = NpyCapsule_AsVoidPtr(attr);
    if (inter->two != 2) {
        goto fail;
    }
    if ((inter->flags & NPY_NOTSWAPPED) != NPY_NOTSWAPPED) {
        endian = PyArray_OPPBYTE;
        inter->flags &= ~NPY_NOTSWAPPED;
    }

    if (inter->flags & ARR_HAS_DESCR) {
        if (PyArray_DescrConverter(inter->descr, &thetype) == PY_FAIL) {
            thetype = NULL;
            PyErr_Clear();
        }
    }

    if (thetype == NULL) {
        PyOS_snprintf(buf, sizeof(buf),
                "%c%c%d", endian, inter->typekind, inter->itemsize);
        if (!(thetype=_array_typedescr_fromstr(buf))) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    r = PyArray_NewFromDescr(&PyArray_Type, thetype,
                             inter->nd, inter->shape,
                             inter->strides, inter->data,
                             inter->flags, NULL);
    Py_INCREF(input);
    PyArray_BASE(r) = input;
    Py_DECREF(attr);
    PyArray_UpdateFlags((PyArrayObject *)r, UPDATE_ALL);
    return r;

 fail:
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    Py_DECREF(attr);
    return NULL;
}

#define PyIntOrLong_Check(obj) (PyInt_Check(obj) || PyLong_Check(obj))

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input)
{
    PyObject *attr = NULL, *item = NULL;
    PyObject *tstr = NULL, *shape = NULL;
    PyObject *inter = NULL;
    PyObject *base = NULL;
    PyArrayObject *ret;
    PyArray_Descr *type=NULL;
    char *data;
    Py_ssize_t buffer_len;
    int res, i, n;
    intp dims[MAX_DIMS], strides[MAX_DIMS];
    int dataflags = BEHAVED;

    /* Get the memory from __array_data__ and __array_offset__ */
    /* Get the shape */
    /* Get the typestring -- ignore array_descr */
    /* Get the strides */

    inter = PyObject_GetAttrString(input, "__array_interface__");
    if (inter == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!PyDict_Check(inter)) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    shape = PyDict_GetItemString(inter, "shape");
    if (shape == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    tstr = PyDict_GetItemString(inter, "typestr");
    if (tstr == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }

    attr = PyDict_GetItemString(inter, "data");
    base = input;
    if ((attr == NULL) || (attr==Py_None) || (!PyTuple_Check(attr))) {
        if (attr && (attr != Py_None)) {
            item = attr;
        }
        else {
            item = input;
        }
        res = PyObject_AsWriteBuffer(item, (void **)&data, &buffer_len);
        if (res < 0) {
            PyErr_Clear();
            res = PyObject_AsReadBuffer(
                    item, (const void **)&data, &buffer_len);
            if (res < 0) {
                goto fail;
            }
            dataflags &= ~NPY_WRITEABLE;
        }
        attr = PyDict_GetItemString(inter, "offset");
        if (attr) {
            longlong num = PyLong_AsLongLong(attr);
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                        "__array_interface__ offset must be an integer");
                goto fail;
            }
            data += num;
        }
        base = item;
    }
    else {
        PyObject *dataptr;
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ data must be a 2-tuple with "
                    "(data pointer integer, read-only flag)");
            goto fail;
        }
        dataptr = PyTuple_GET_ITEM(attr, 0);
        if (PyString_Check(dataptr)) {
            res = sscanf(PyString_AsString(dataptr),
                         "%p", (void **)&data);
            if (res < 1) {
                PyErr_SetString(PyExc_TypeError,
                        "__array_interface__ data string cannot be converted");
                goto fail;
            }
        }
        else if (PyIntOrLong_Check(dataptr)) {
            data = PyLong_AsVoidPtr(dataptr);
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "first element of __array_interface__ data tuple "
                    "must be integer or string.");
            goto fail;
        }
        if (PyObject_IsTrue(PyTuple_GET_ITEM(attr,1))) {
            dataflags &= ~NPY_WRITEABLE;
        }
    }
    attr = tstr;
#if defined(NPY_PY3K)
    if (PyUnicode_Check(tstr)) {
        /* Allow unicode type strings */
        attr = PyUnicode_AsASCIIString(tstr);
    }
#endif
    if (!PyBytes_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "typestr must be a string");
        goto fail;
    }
    type = _array_typedescr_fromstr(PyString_AS_STRING(attr));
#if defined(NPY_PY3K)
    if (attr != tstr) {
        Py_DECREF(attr);
    }
#endif
    if (type == NULL) {
        goto fail;
    }
    attr = shape;
    if (!PyTuple_Check(attr)) {
        PyErr_SetString(PyExc_TypeError,
                "shape must be a tuple");
        Py_DECREF(type);
        goto fail;
    }
    n = PyTuple_GET_SIZE(attr);
    for (i = 0; i < n; i++) {
        item = PyTuple_GET_ITEM(attr, i);
        dims[i] = PyArray_PyIntAsIntp(item);
        if (error_converting(dims[i])) {
            break;
        }
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, type,
                                                n, dims,
                                                NULL, data,
                                                dataflags, NULL);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(base);
    ret->base = base;

    attr = PyDict_GetItemString(inter, "strides");
    if (attr != NULL && attr != Py_None) {
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                    "strides must be a tuple");
            Py_DECREF(ret);
            return NULL;
        }
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                    "mismatch in length of strides and shape");
            Py_DECREF(ret);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            item = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(item);
            if (error_converting(strides[i])) {
                break;
            }
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        memcpy(ret->strides, strides, n*sizeof(npy_intp));
    }
    else PyErr_Clear();
    PyArray_UpdateFlags(ret, UPDATE_ALL);
    Py_DECREF(inter);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(inter);
    return NULL;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    PyObject *new;
    PyObject *array_meth;

    array_meth = PyObject_GetAttrString(op, "__array__");
    if (array_meth == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (context == NULL) {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, NULL);
        }
        else {
            new = PyObject_CallFunction(array_meth, "O", typecode);
        }
    }
    else {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, "OO", Py_None, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "");
            }
        }
        else {
            new = PyObject_CallFunction(array_meth, "OO", typecode, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "O", typecode);
            }
        }
    }
    Py_DECREF(array_meth);
    if (new == NULL) {
        return NULL;
    }
    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not "  \
                        "producing an array");
        Py_DECREF(new);
        return NULL;
    }
    return new;
}

/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
{
    return _array_find_type(op, mintype, MAX_DIMS);
}

/* These are also old calls (should use PyArray_NewFromDescr) */

/* They all zero-out the memory as previously done */

/* steals reference to descr -- and enforces native byteorder on it.*/
/*NUMPY_API
  Like FromDimsAndData but uses the Descr structure instead of typecode
  as input.
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDimsAndDataAndDescr(int nd, int *d,
                                PyArray_Descr *descr,
                                char *data)
{
    PyObject *ret;
    int i;
    npy_intp newd[MAX_DIMS];
    char msg[] = "PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    if (!PyArray_ISNBO(descr->byteorder))
        descr->byteorder = '=';
    for (i = 0; i < nd; i++) {
        newd[i] = (npy_intp) d[i];
    }
    ret = PyArray_NewFromDescr(&PyArray_Type, descr,
                               nd, newd,
                               NULL, data,
                               (data ? CARRAY : 0), NULL);
    return ret;
}

/*NUMPY_API
  Construct an empty array from dimensions and typenum
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDims(int nd, int *d, int type)
{
    PyObject *ret;
    char msg[] = "PyArray_FromDims: use PyArray_SimpleNew.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    ret = PyArray_FromDimsAndDataAndDescr(nd, d,
                                          PyArray_DescrFromType(type),
                                          NULL);
    /*
     * Old FromDims set memory to zero --- some algorithms
     * relied on that.  Better keep it the same. If
     * Object type, then it's already been set to zero, though.
     */
    if (ret && (PyArray_DESCR(ret)->type_num != PyArray_OBJECT)) {
        memset(PyArray_DATA(ret), 0, PyArray_NBYTES(ret));
    }
    return ret;
}

/* end old calls */

/*NUMPY_API
 * This is a quick wrapper around PyArray_FromAny(op, NULL, 0, 0, ENSUREARRAY)
 * that special cases Arrays and PyArray_Scalars up front
 * It *steals a reference* to the object
 * It also guarantees that the result is PyArray_Type
 * Because it decrefs op if any conversion needs to take place
 * so it can be used like PyArray_EnsureArray(some_function(...))
 */
NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op)
{
    PyObject *new;

    if ((op == NULL) || (PyArray_CheckExact(op))) {
        new = op;
        Py_XINCREF(new);
    }
    else if (PyArray_Check(op)) {
        new = PyArray_View((PyArrayObject *)op, NULL, &PyArray_Type);
    }
    else if (PyArray_IsScalar(op, Generic)) {
        new = PyArray_FromScalar(op, NULL);
    }
    else {
        new = PyArray_FromAny(op, NULL, 0, 0, NPY_ENSUREARRAY, NULL);
    }
    Py_XDECREF(op);
    return new;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op)
{
    if (op && PyArray_Check(op)) {
        return op;
    }
    return PyArray_EnsureArray(op);
}

/* TODO: Put the order parameter in PyArray_CopyAnyInto and remove this */
NPY_NO_EXPORT int
PyArray_CopyAnyIntoOrdered(PyArrayObject *dst, PyArrayObject *src,
                                NPY_ORDER order)
{
    PyArray_StridedTransferFn *stransfer = NULL;
    void *transferdata = NULL;
    NpyIter *dst_iter, *src_iter;

    NpyIter_IterNextFunc *dst_iternext, *src_iternext;
    char **dst_dataptr, **src_dataptr;
    npy_intp dst_stride, src_stride;
    npy_intp *dst_countptr, *src_countptr;

    char *dst_data, *src_data;
    npy_intp dst_count, src_count, count;
    npy_intp src_itemsize;
    npy_intp dst_size, src_size;
    int needs_api;

    NPY_BEGIN_THREADS_DEF;

    if (!PyArray_ISWRITEABLE(dst)) {
        PyErr_SetString(PyExc_RuntimeError,
                "cannot write to array");
        return -1;
    }

    /*
     * If the shapes match and a particular order is forced
     * for both, use the more efficient CopyInto
     */
    if (order != NPY_ANYORDER && order != NPY_KEEPORDER &&
            PyArray_NDIM(dst) == PyArray_NDIM(src) &&
            PyArray_CompareLists(PyArray_DIMS(dst), PyArray_DIMS(src),
                                PyArray_NDIM(dst))) {
        return PyArray_CopyInto(dst, src);
    }

    dst_size = PyArray_SIZE(dst);
    src_size = PyArray_SIZE(src);
    if (dst_size != src_size) {
        PyErr_SetString(PyExc_ValueError,
                "arrays must have the same number of elements"
                " for copy");
        return -1;
    }

    /* Zero-sized arrays require nothing be done */
    if (dst_size == 0) {
        return 0;
    }


    /*
     * This copy is based on matching C-order traversals of src and dst.
     * By using two iterators, we can find maximal sub-chunks that
     * can be processed at once.
     */
    dst_iter = NpyIter_New(dst, NPY_ITER_WRITEONLY|
                                NPY_ITER_EXTERNAL_LOOP|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_REFS_OK,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (dst_iter == NULL) {
        return -1;
    }
    src_iter = NpyIter_New(src, NPY_ITER_READONLY|
                                NPY_ITER_EXTERNAL_LOOP|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_REFS_OK,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (src_iter == NULL) {
        NpyIter_Deallocate(dst_iter);
        return -1;
    }

    /* Get all the values needed for the inner loop */
    dst_iternext = NpyIter_GetIterNext(dst_iter, NULL);
    dst_dataptr = NpyIter_GetDataPtrArray(dst_iter);
    /* Since buffering is disabled, we can cache the stride */
    dst_stride = *NpyIter_GetInnerStrideArray(dst_iter);
    dst_countptr = NpyIter_GetInnerLoopSizePtr(dst_iter);

    src_iternext = NpyIter_GetIterNext(src_iter, NULL);
    src_dataptr = NpyIter_GetDataPtrArray(src_iter);
    /* Since buffering is disabled, we can cache the stride */
    src_stride = *NpyIter_GetInnerStrideArray(src_iter);
    src_countptr = NpyIter_GetInnerLoopSizePtr(src_iter);

    if (dst_iternext == NULL || src_iternext == NULL) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }

    src_itemsize = PyArray_DESCR(src)->elsize;

    needs_api = NpyIter_IterationNeedsAPI(dst_iter) ||
                NpyIter_IterationNeedsAPI(src_iter);

    /*
     * Because buffering is disabled in the iterator, the inner loop
     * strides will be the same throughout the iteration loop.  Thus,
     * we can pass them to this function to take advantage of
     * contiguous strides, etc.
     */
    if (PyArray_GetDTypeTransferFunction(
                    PyArray_ISALIGNED(src) && PyArray_ISALIGNED(dst),
                    src_stride, dst_stride,
                    PyArray_DESCR(src), PyArray_DESCR(dst),
                    0,
                    &stransfer, &transferdata,
                    &needs_api) != NPY_SUCCEED) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }


    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    dst_count = *dst_countptr;
    src_count = *src_countptr;
    dst_data = *dst_dataptr;
    src_data = *src_dataptr;
    /*
     * The tests did not trigger this code, so added a new function
     * ndarray.setasflat to the Python exposure in order to test it.
     */
    for(;;) {
        /* Transfer the biggest amount that fits both */
        count = (src_count < dst_count) ? src_count : dst_count;
        stransfer(dst_data, dst_stride,
                    src_data, src_stride,
                    count, src_itemsize, transferdata);

        /* If we exhausted the dst block, refresh it */
        if (dst_count == count) {
            if (!dst_iternext(dst_iter)) {
                break;
            }
            dst_count = *dst_countptr;
            dst_data = *dst_dataptr;
        }
        else {
            dst_count -= count;
            dst_data += count*dst_stride;
        }

        /* If we exhausted the src block, refresh it */
        if (src_count == count) {
            if (!src_iternext(src_iter)) {
                break;
            }
            src_count = *src_countptr;
            src_data = *src_dataptr;
        }
        else {
            src_count -= count;
            src_data += count*src_stride;
        }
    }

    if (!needs_api) {
        NPY_END_THREADS;
    }

    PyArray_FreeStridedTransferData(transferdata);
    NpyIter_Deallocate(dst_iter);
    NpyIter_Deallocate(src_iter);

    return PyErr_Occurred() ? -1 : 0;
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 *
 * TODO: For NumPy 2.0, this could accept an order parameter which
 *       only allows NPY_CORDER and NPY_FORDER.  Could also rename
 *       this to CopyAsFlat to make the name more intuitive.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dst, PyArrayObject *src)
{
    return PyArray_CopyAnyIntoOrdered(dst, src, NPY_CORDER);
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dst, PyArrayObject *src)
{
    PyArray_StridedTransferFn *stransfer = NULL;
    void *transferdata = NULL;
    NPY_BEGIN_THREADS_DEF;

    if (!PyArray_ISWRITEABLE(dst)) {
        PyErr_SetString(PyExc_RuntimeError,
                "cannot write to array");
        return -1;
    }

    if (PyArray_NDIM(dst) >= PyArray_NDIM(src) &&
                            PyArray_TRIVIALLY_ITERABLE_PAIR(dst, src)) {
        char *dst_data, *src_data;
        npy_intp count, dst_stride, src_stride, src_itemsize;

        int needs_api = 0;

        PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(dst, src, count,
                              dst_data, src_data, dst_stride, src_stride);

        /*
         * Check for overlap with positive strides, and if found,
         * possibly reverse the order
         */
        if (dst_data > src_data && src_stride > 0 && dst_stride > 0 &&
                        (dst_data < src_data+src_stride*count) &&
                        (src_data < dst_data+dst_stride*count)) {
            dst_data += dst_stride*(count-1);
            src_data += src_stride*(count-1);
            dst_stride = -dst_stride;
            src_stride = -src_stride;
        }

        if (PyArray_GetDTypeTransferFunction(
                        PyArray_ISALIGNED(src) && PyArray_ISALIGNED(dst),
                        src_stride, dst_stride,
                        PyArray_DESCR(src), PyArray_DESCR(dst),
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
            return -1;
        }

        src_itemsize = PyArray_DESCR(src)->elsize;

        if (!needs_api) {
            NPY_BEGIN_THREADS;
        }

        stransfer(dst_data, dst_stride, src_data, src_stride,
                    count, src_itemsize, transferdata);

        if (!needs_api) {
            NPY_END_THREADS;
        }

        PyArray_FreeStridedTransferData(transferdata);

        return PyErr_Occurred() ? -1 : 0;
    }
    else {
        PyArrayObject *op[2];
        npy_uint32 op_flags[2];
        NpyIter *iter;

        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *stride;
        npy_intp *countptr;
        npy_intp src_itemsize;
        int needs_api;

        op[0] = dst;
        op[1] = src;
        /*
         * TODO: In NumPy 2.0, renable NPY_ITER_NO_BROADCAST. This
         *       was removed during NumPy 1.6 testing for compatibility
         *       with NumPy 1.5, as per Travis's -10 veto power.
         */
        /*op_flags[0] = NPY_ITER_WRITEONLY|NPY_ITER_NO_BROADCAST;*/
        op_flags[0] = NPY_ITER_WRITEONLY;
        op_flags[1] = NPY_ITER_READONLY;

        iter = NpyIter_MultiNew(2, op,
                            NPY_ITER_EXTERNAL_LOOP|
                            NPY_ITER_REFS_OK|
                            NPY_ITER_ZEROSIZE_OK,
                            NPY_KEEPORDER,
                            NPY_NO_CASTING,
                            op_flags,
                            NULL);
        if (iter == NULL) {
            return -1;
        }

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        stride = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);
        src_itemsize = PyArray_DESCR(src)->elsize;

        needs_api = NpyIter_IterationNeedsAPI(iter);

        /*
         * Because buffering is disabled in the iterator, the inner loop
         * strides will be the same throughout the iteration loop.  Thus,
         * we can pass them to this function to take advantage of
         * contiguous strides, etc.
         */
        if (PyArray_GetDTypeTransferFunction(
                        PyArray_ISALIGNED(src) && PyArray_ISALIGNED(dst),
                        stride[1], stride[0],
                        PyArray_DESCR(src), PyArray_DESCR(dst),
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            return -1;
        }


        if (NpyIter_GetIterSize(iter) != 0) {
            if (!needs_api) {
                NPY_BEGIN_THREADS;
            }

            do {
                stransfer(dataptr[0], stride[0],
                            dataptr[1], stride[1],
                            *countptr, src_itemsize, transferdata);
            } while(iternext(iter));

            if (!needs_api) {
                NPY_END_THREADS;
            }
        }

        PyArray_FreeStridedTransferData(transferdata);
        NpyIter_Deallocate(iter);

        return PyErr_Occurred() ? -1 : 0;
    }
}


/*NUMPY_API
  PyArray_CheckAxis

  check that axis is valid
  convert 0-d arrays to 1-d arrays
*/
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    PyObject *temp1, *temp2;
    int n = arr->nd;

    if (*axis == MAX_DIMS || n == 0) {
        if (n != 1) {
            temp1 = PyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            if (*axis == MAX_DIMS) {
                *axis = PyArray_NDIM(temp1)-1;
            }
        }
        else {
            temp1 = (PyObject *)arr;
            Py_INCREF(temp1);
            *axis = 0;
        }
        if (!flags && *axis == 0) {
            return temp1;
        }
    }
    else {
        temp1 = (PyObject *)arr;
        Py_INCREF(temp1);
    }
    if (flags) {
        temp2 = PyArray_CheckFromAny((PyObject *)temp1, NULL,
                                     0, 0, flags, NULL);
        Py_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    }
    else {
        temp2 = (PyObject *)temp1;
    }
    n = PyArray_NDIM(temp2);
    if (*axis < 0) {
        *axis += n;
    }
    if ((*axis < 0) || (*axis >= n)) {
        PyErr_Format(PyExc_ValueError,
                     "axis(=%d) out of bounds", *axis);
        Py_DECREF(temp2);
        return NULL;
    }
    return temp2;
}

/*NUMPY_API
 * Zeros
 *
 * steal a reference
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros(int nd, npy_intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    if (!type) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type,
                                                nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (_zerofill(ret) < 0) {
        return NULL;
    }
    return (PyObject *)ret;

}

/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals referenct to type
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty(int nd, npy_intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    if (!type) type = PyArray_DescrFromType(PyArray_DEFAULT);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type, nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (PyDataType_REFCHK(type)) {
        PyArray_FillObjectArray(ret, Py_None);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return NULL;
        }
    }
    return (PyObject *)ret;
}

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static int _safe_ceil_to_intp(double value, npy_intp* ret)
{
    double ivalue;

    ivalue = npy_ceil(value);
    if (ivalue < NPY_MIN_INTP || ivalue > NPY_MAX_INTP) {
        return -1;
    }

    *ret = (npy_intp)ivalue;
    return 0;
}


/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    npy_intp length;
    PyObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *obj;
    int ret;

    if (_safe_ceil_to_intp((stop - start)/step, &length)) {
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);
    }
    range = PyArray_New(&PyArray_Type, 1, &length, type_num,
                        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyArray_DESCR(range)->f;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_DATA(range), (PyArrayObject *)range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                         (PyArrayObject *)range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    funcs->fill(PyArray_DATA(range), length, (PyArrayObject *)range);
    if (PyErr_Occurred()) {
        goto fail;
    }
    return range;

 fail:
    Py_DECREF(range);
    return NULL;
}

/*
 * the formula is len = (intp) ceil((start - stop) / step);
 */
static npy_intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    npy_intp len, tmp;
    PyObject *val;
    double value;

    *next = PyNumber_Subtract(stop, start);
    if (!(*next)) {
        if (PyTuple_Check(stop)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");
        }
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;
    if (!val) {
        return -1;
    }
    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            Py_DECREF(val);
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &tmp)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        len = MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
    }
    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!next) {
            return -1;
        }
    }
    return len;
}

/*NUMPY_API
 *
 * ArangeObj,
 *
 * this doesn't change the references
 */
NPY_NO_EXPORT PyObject *
PyArray_ArangeObj(PyObject *start, PyObject *stop, PyObject *step, PyArray_Descr *dtype)
{
    PyObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *next, *err;
    npy_intp length;
    PyArray_Descr *native = NULL;
    int swap;

    if (!dtype) {
        PyArray_Descr *deftype;
        PyArray_Descr *newtype;

        /* intentionally made to be at least NPY_LONG */
        deftype = PyArray_DescrFromType(NPY_LONG);
        newtype = PyArray_DescrFromObject(start, deftype);
        Py_DECREF(deftype);
        if (newtype == NULL) {
            return NULL;
        }
        deftype = newtype;
        if (stop && stop != Py_None) {
            newtype = PyArray_DescrFromObject(stop, deftype);
            Py_DECREF(deftype);
            if (newtype == NULL) {
                return NULL;
            }
            deftype = newtype;
        }
        if (step && step != Py_None) {
            newtype = PyArray_DescrFromObject(step, deftype);
            Py_DECREF(deftype);
            if (newtype == NULL) {
                return NULL;
            }
            deftype = newtype;
        }
        dtype = deftype;
    }
    else {
        Py_INCREF(dtype);
    }
    if (!step || step == Py_None) {
        step = PyInt_FromLong(1);
    }
    else {
        Py_XINCREF(step);
    }
    if (!stop || stop == Py_None) {
        stop = start;
        start = PyInt_FromLong(0);
    }
    else {
        Py_INCREF(start);
    }
    /* calculate the length and next = start + step*/
    length = _calc_length(start, stop, step, &next,
                          PyTypeNum_ISCOMPLEX(dtype->type_num));
    err = PyErr_Occurred();
    if (err) {
        Py_DECREF(dtype);
        if (err && PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    if (length <= 0) {
        length = 0;
        range = PyArray_SimpleNewFromDescr(1, &length, dtype);
        Py_DECREF(step);
        Py_DECREF(start);
        return range;
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    if (!PyArray_ISNBO(dtype->byteorder)) {
        native = PyArray_DescrNewByteorder(dtype, PyArray_NATBYTE);
        swap = 1;
    }
    else {
        native = dtype;
        swap = 0;
    }

    range = PyArray_SimpleNewFromDescr(1, &length, native);
    if (range == NULL) {
        goto fail;
    }

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    funcs = PyArray_DESCR(range)->f;
    if (funcs->setitem(
                start, PyArray_DATA(range), (PyArrayObject *)range) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       (PyArrayObject *)range) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        goto fail;
    }
    funcs->fill(PyArray_DATA(range), length, (PyArrayObject *)range);
    if (PyErr_Occurred()) {
        goto fail;
    }
 finish:
    if (swap) {
        PyObject *new;
        new = PyArray_Byteswap((PyArrayObject *)range, 1);
        Py_DECREF(new);
        Py_DECREF(PyArray_DESCR(range));
        PyArray_DESCR(range) = dtype;  /* steals the reference */
    }
    Py_DECREF(start);
    Py_DECREF(step);
    Py_DECREF(next);
    return range;

 fail:
    Py_DECREF(start);
    Py_DECREF(step);
    Py_XDECREF(next);
    return NULL;
}

static PyArrayObject *
array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, npy_intp num, size_t *nread)
{
    PyArrayObject *r;
    npy_intp start, numbytes;

    if (num < 0) {
        int fail = 0;

#if defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400)
        /* Workaround Win64 fwrite() bug. Ticket #1660 */
        start = (npy_intp )_ftelli64(fp);
        if (start < 0) {
            fail = 1;
        }
        if (_fseeki64(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = (npy_intp) _ftelli64(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (_fseeki64(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
#else
        start = (npy_intp)ftell(fp);
        if (start < 0) {
            fail = 1;
        }
        if (fseek(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = (npy_intp) ftell(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (fseek(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
#endif
        if (fail) {
            PyErr_SetString(PyExc_IOError,
                            "could not seek in file");
            Py_DECREF(dtype);
            return NULL;
        }
        num = numbytes / dtype->elsize;
    }
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                              dtype,
                                              1, &num,
                                              NULL, NULL,
                                              0, NULL);
    if (r == NULL) {
        return NULL;
    }
    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(r->data, dtype->elsize, num, fp);
    NPY_END_ALLOW_THREADS;
    return r;
}

/*
 * Create an array by reading from the given stream, using the passed
 * next_element and skip_separator functions.
 */
#define FROM_BUFFER_SIZE 4096
static PyArrayObject *
array_from_text(PyArray_Descr *dtype, npy_intp num, char *sep, size_t *nread,
                void *stream, next_element next, skip_separator skip_sep,
                void *stream_data)
{
    PyArrayObject *r;
    npy_intp i;
    char *dptr, *clean_sep, *tmp;
    int err = 0;
    npy_intp thisbuf = 0;
    npy_intp size;
    npy_intp bytes, totalbytes;

    size = (num >= 0) ? num : FROM_BUFFER_SIZE;
    r = (PyArrayObject *)
        PyArray_NewFromDescr(&PyArray_Type,
                             dtype,
                             1, &size,
                             NULL, NULL,
                             0, NULL);
    if (r == NULL) {
        return NULL;
    }
    clean_sep = swab_separator(sep);
    NPY_BEGIN_ALLOW_THREADS;
    totalbytes = bytes = size * dtype->elsize;
    dptr = r->data;
    for (i= 0; num < 0 || i < num; i++) {
        if (next(&stream, dptr, dtype, stream_data) < 0) {
            break;
        }
        *nread += 1;
        thisbuf += 1;
        dptr += dtype->elsize;
        if (num < 0 && thisbuf == size) {
            totalbytes += bytes;
            tmp = PyDataMem_RENEW(r->data, totalbytes);
            if (tmp == NULL) {
                err = 1;
                break;
            }
            r->data = tmp;
            dptr = tmp + (totalbytes - bytes);
            thisbuf = 0;
        }
        if (skip_sep(&stream, clean_sep, stream_data) < 0) {
            break;
        }
    }
    if (num < 0) {
        tmp = PyDataMem_RENEW(r->data, NPY_MAX(*nread,1)*dtype->elsize);
        if (tmp == NULL) {
            err = 1;
        }
        else {
            PyArray_DIM(r,0) = *nread;
            r->data = tmp;
        }
    }
    NPY_END_ALLOW_THREADS;
    free(clean_sep);
    if (err == 1) {
        PyErr_NoMemory();
    }
    if (PyErr_Occurred()) {
        Py_DECREF(r);
        return NULL;
    }
    return r;
}
#undef FROM_BUFFER_SIZE

/*NUMPY_API
 *
 * Given a ``FILE *`` pointer ``fp``, and a ``PyArray_Descr``, return an
 * array corresponding to the data encoded in that file.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 *
 * For memory-mapped files, use the buffer interface. No more data than
 * necessary is read by this routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *dtype, npy_intp num, char *sep)
{
    PyArrayObject *ret;
    size_t nread = 0;

    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                "The elements are 0-sized.");
        Py_DECREF(dtype);
        return NULL;
    }
    if ((sep == NULL) || (strlen(sep) == 0)) {
        ret = array_fromfile_binary(fp, dtype, num, &nread);
    }
    else {
        if (dtype->f->scanfunc == NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Unable to read character files of that array type");
            Py_DECREF(dtype);
            return NULL;
        }
        ret = array_from_text(dtype, num, sep, &nread, fp,
                (next_element) fromfile_next_element,
                (skip_separator) fromfile_skip_separator, NULL);
    }
    if (ret == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }
    if (((npy_intp) nread) < num) {
        /* Realloc memory for smaller number of elements */
        const size_t nsize = NPY_MAX(nread,1)*ret->descr->elsize;
        char *tmp;

        if((tmp = PyDataMem_RENEW(ret->data, nsize)) == NULL) {
            Py_DECREF(ret);
            return PyErr_NoMemory();
        }
        ret->data = tmp;
        PyArray_DIM(ret,0) = nread;
    }
    return (PyObject *)ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   npy_intp count, npy_intp offset)
{
    PyArrayObject *ret;
    char *data;
    Py_ssize_t ts;
    npy_intp s, n;
    int itemsize;
    int writeable = 1;


    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory"\
                        " buffer");
        Py_DECREF(type);
        return NULL;
    }
    if (type->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;
    }
    if (Py_TYPE(buf)->tp_as_buffer == NULL
#if defined(NPY_PY3K)
        || Py_TYPE(buf)->tp_as_buffer->bf_getbuffer == NULL
#else
        || (Py_TYPE(buf)->tp_as_buffer->bf_getwritebuffer == NULL
            && Py_TYPE(buf)->tp_as_buffer->bf_getreadbuffer == NULL)
#endif
        ) {
        PyObject *newbuf;
        newbuf = PyObject_GetAttrString(buf, "__buffer__");
        if (newbuf == NULL) {
            Py_DECREF(type);
            return NULL;
        }
        buf = newbuf;
    }
    else {
        Py_INCREF(buf);
    }

    if (PyObject_AsWriteBuffer(buf, (void *)&data, &ts) == -1) {
        writeable = 0;
        PyErr_Clear();
        if (PyObject_AsReadBuffer(buf, (void *)&data, &ts) == -1) {
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((offset < 0) || (offset > ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be non-negative and no greater than buffer "\
                     "length (%" INTP_FMT ")", (npy_intp)ts);
        Py_DECREF(buf);
        Py_DECREF(type);
        return NULL;
    }

    data += offset;
    s = (npy_intp)ts - offset;
    n = (npy_intp)count;
    itemsize = type->elsize;
    if (n < 0 ) {
        if (s % itemsize != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size must be a multiple"\
                            " of element size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
        n = s/itemsize;
    }
    else {
        if (s < n*itemsize) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer is smaller than requested"\
                            " size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                     type,
                                                     1, &n,
                                                     NULL, data,
                                                     DEFAULT,
                                                     NULL)) == NULL) {
        Py_DECREF(buf);
        return NULL;
    }

    if (!writeable) {
        ret->flags &= ~NPY_WRITEABLE;
    }
    /* Store a reference for decref on deallocation */
    ret->base = buf;
    PyArray_UpdateFlags(ret, NPY_ALIGNED);
    return (PyObject *)ret;
}

/*NUMPY_API
 *
 * Given a pointer to a string ``data``, a string length ``slen``, and
 * a ``PyArray_Descr``, return an array corresponding to the data
 * encoded in that string.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * If ``slen`` is < 0, then the end of string is used for text data.
 * It is an error for ``slen`` to be < 0 for binary data (since embedded NULLs
 * would be the norm).
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromString(char *data, npy_intp slen, PyArray_Descr *dtype,
                   npy_intp num, char *sep)
{
    int itemsize;
    PyArrayObject *ret;
    Bool binary;

    if (dtype == NULL) {
        dtype=PyArray_DescrFromType(PyArray_DEFAULT);
    }
    if (PyDataType_FLAGCHK(dtype, NPY_ITEM_IS_POINTER) ||
                    PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot create an object array from"    \
                        " a string");
        Py_DECREF(dtype);
        return NULL;
    }
    itemsize = dtype->elsize;
    if (itemsize == 0) {
        PyErr_SetString(PyExc_ValueError, "zero-valued itemsize");
        Py_DECREF(dtype);
        return NULL;
    }

    binary = ((sep == NULL) || (strlen(sep) == 0));
    if (binary) {
        if (num < 0 ) {
            if (slen % itemsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                                "string size must be a "\
                                "multiple of element size");
                Py_DECREF(dtype);
                return NULL;
            }
            num = slen/itemsize;
        }
        else {
            if (slen < num*itemsize) {
                PyErr_SetString(PyExc_ValueError,
                                "string is smaller than " \
                                "requested size");
                Py_DECREF(dtype);
                return NULL;
            }
        }
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, dtype,
                                 1, &num, NULL, NULL,
                                 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
        memcpy(ret->data, data, num*dtype->elsize);
    }
    else {
        /* read from character-based string */
        size_t nread = 0;
        char *end;

        if (dtype->f->scanfunc == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "don't know how to read "       \
                            "character strings with that "  \
                            "array type");
            Py_DECREF(dtype);
            return NULL;
        }
        if (slen < 0) {
            end = NULL;
        }
        else {
            end = data + slen;
        }
        ret = array_from_text(dtype, num, sep, &nread,
                              data,
                              (next_element) fromstr_next_element,
                              (skip_separator) fromstr_skip_separator,
                              end);
    }
    return (PyObject *)ret;
}

/*NUMPY_API
 *
 * steals a reference to dtype (which cannot be NULL)
 */
NPY_NO_EXPORT PyObject *
PyArray_FromIter(PyObject *obj, PyArray_Descr *dtype, npy_intp count)
{
    PyObject *value;
    PyObject *iter = PyObject_GetIter(obj);
    PyArrayObject *ret = NULL;
    npy_intp i, elsize, elcount;
    char *item, *new_data;

    if (iter == NULL) {
        goto done;
    }
    elcount = (count < 0) ? 0 : count;
    if ((elsize=dtype->elsize) == 0) {
        PyErr_SetString(PyExc_ValueError, "Must specify length "\
                        "when using variable-size data-type.");
        goto done;
    }

    /*
     * We would need to alter the memory RENEW code to decrement any
     * reference counts before throwing away any memory.
     */
    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError, "cannot create "\
                        "object arrays from iterator");
        goto done;
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1,
                                                &elcount, NULL,NULL, 0, NULL);
    dtype = NULL;
    if (ret == NULL) {
        goto done;
    }
    for (i = 0; (i < count || count == -1) &&
             (value = PyIter_Next(iter)); i++) {
        if (i >= elcount) {
            /*
              Grow ret->data:
              this is similar for the strategy for PyListObject, but we use
              50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            if (elcount <= NPY_MAX_INTP/elsize) {
                new_data = PyDataMem_RENEW(ret->data, elcount * elsize);
            }
            else {
                new_data = NULL;
            }
            if (new_data == NULL) {
                PyErr_SetString(PyExc_MemoryError,
                                "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            ret->data = new_data;
        }
        ret->dimensions[0] = i + 1;

        if (((item = index2ptr(ret, i)) == NULL)
            || (ret->descr->f->setitem(value, item, ret) == -1)) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }

    if (i < count) {
        PyErr_SetString(PyExc_ValueError, "iterator too short");
        goto done;
    }

    /*
     * Realloc the data so that don't keep extra memory tied up
     * (assuming realloc is reasonably good about reusing space...)
     */
    if (i == 0) {
        i = 1;
    }
    new_data = PyDataMem_RENEW(ret->data, i * elsize);
    if (new_data == NULL) {
        PyErr_SetString(PyExc_MemoryError, "cannot allocate array memory");
        goto done;
    }
    ret->data = new_data;

 done:
    Py_XDECREF(iter);
    Py_XDECREF(dtype);
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the NPY_C_CONTIGUOUS bit will be set.  If the flags argument
 * has the NPY_F_CONTIGUOUS bit set, then a FORTRAN-style strides array will be
 * created (and of course the NPY_F_CONTIGUOUS flag bit will be set).
 *
 * If data is not given but created here, then flags will be DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 */

NPY_NO_EXPORT size_t
_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & (NPY_F_CONTIGUOUS|NPY_C_CONTIGUOUS)) == NPY_F_CONTIGUOUS) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        if (nd > 1) {
            *objflags = ((*objflags)|NPY_F_CONTIGUOUS) & ~NPY_C_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_F_CONTIGUOUS|NPY_C_CONTIGUOUS);
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        if (nd > 1) {
            *objflags = ((*objflags)|NPY_C_CONTIGUOUS) & ~NPY_F_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_C_CONTIGUOUS|NPY_F_CONTIGUOUS);
        }
    }
    return itemsize;
}

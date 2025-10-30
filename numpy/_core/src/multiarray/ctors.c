#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_ctypes.h"

#include "npy_static_data.h"

#include "common.h"
#include "ctors.h"
#include "convert_datatype.h"
#include "descriptor.h"
#include "dtypemeta.h"
#include "refcount.h"  /* for PyArray_SetObjectsToNone */
#include "shape.h"
#include "npy_buffer.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_assign.h"
#include "mapping.h" /* for array_item_asarray */
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "alloc.h"
#include <assert.h>

#include "get_attr_string.h"
#include "array_coercion.h"

#include "umathmodule.h"


NPY_NO_EXPORT const char *npy_no_copy_err_msg = (
        "Unable to avoid copy while creating an array as requested.\n"
        "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` "
        "to allow a copy when needed (no behavior change in NumPy 1.x).\n"
        "For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.");

/*
 * Reading from a file or a string.
 *
 * As much as possible, we try to use the same code for both files and strings,
 * so the semantics for fromstring and fromfile are the same, especially with
 * regards to the handling of text representations.
 */

/*
 * Scanning function for next element parsing and separator skipping.
 * These functions return:
 *   - 0 to indicate more data to read
 *   - -1 when reading stopped at the end of the string/file
 *   - -2 when reading stopped before the end was reached.
 *
 * The dtype specific parsing functions may set the python error state
 * (they have to get the GIL first) additionally.
 */
typedef int (*next_element)(void **, void *, PyArray_Descr *, void *);
typedef int (*skip_separator)(void **, const char *, void *);


static npy_bool
string_is_fully_read(char const* start, char const* end) {
    if (end == NULL) {
        return *start == '\0';  /* null terminated */
    }
    else {
        return start >= end;  /* fixed length */
    }
}


static int
fromstr_next_element(char **s, void *dptr, PyArray_Descr *dtype,
                     const char *end)
{
    char *e = *s;
    int r = PyDataType_GetArrFuncs(dtype)->fromstr(*s, dptr, &e, dtype);
    /*
     * fromstr always returns 0 for basic dtypes; s points to the end of the
     * parsed string. If s is not changed an error occurred or the end was
     * reached.
     */
    if (*s == e || r < 0) {
        /* Nothing read, could be end of string or an error (or both) */
        if (string_is_fully_read(*s, end)) {
            return -1;
        }
        return -2;
    }
    *s = e;
    if (end != NULL && *s > end) {
        /* Stop the iteration if we read far enough */
        return -1;
    }
    return 0;
}

static int
fromfile_next_element(FILE **fp, void *dptr, PyArray_Descr *dtype,
                      void *NPY_UNUSED(stream_data))
{
    /* the NULL argument is for backwards-compatibility */
    int r = PyDataType_GetArrFuncs(dtype)->scanfunc(*fp, dptr, NULL, dtype);
    /* r can be EOF or the number of items read (0 or 1) */
    if (r == 1) {
        return 0;
    }
    else if (r == EOF) {
        return -1;
    }
    else {
        /* unable to read more, but EOF not reached indicating an error. */
        return -2;
    }
}

/*
 * Remove multiple whitespace from the separator, and add a space to the
 * beginning and end. This simplifies the separator-skipping code below.
 */
static char *
swab_separator(const char *sep)
{
    int skip_space = 0;
    char *s, *start;

    s = start = malloc(strlen(sep)+3);
    if (s == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
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
        if (string_is_fully_read(string, end)) {
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
    _PyArray_LegacyDescr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    int tuple;

    old = (_PyArray_LegacyDescr *)*des;  /* guaranteed as it has subarray */
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
    if (newnd > NPY_MAXDIMS) {
        goto finish;
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (npy_intp) PyLong_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));
        }
    }
    else {
        mydim[0] = (npy_intp) PyLong_AsLong(old->subarray->shape);
    }

    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;

        mystrides = newstrides + oldnd;
        /* Make new strides -- always C-contiguous */
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

#define _COPY_N_SIZE(size) \
    for(i=0; i<N; i++) { \
        memcpy(tout, tin, size); \
        tin += instrides; \
        tout += outstrides; \
    } \
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
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint32))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint32 * a_ = (npy_uint32 *)a;
                *a_ = npy_bswap4(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap4_unaligned(a);
            }
        }
        break;
    case 8:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint64))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint64 * a_ = (npy_uint64 *)a;
                *a_ = npy_bswap8(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap8_unaligned(a);
            }
        }
        break;
    case 2:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint16))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint16 * a_ = (npy_uint16 *)a;
                *a_ = npy_bswap2(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap2_unaligned(a);
            }
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
    if ((numitems == 1) || (itemsize == srcstrides)) {
        memcpy(dst, src, itemsize*numitems);
    }
    else {
        npy_intp i;
        char *s1 = (char *)src;
        char *d1 = (char *)dst;

        for (i = 0; i < numitems; i++) {
            memcpy(d1, s1, itemsize);
            d1 += itemsize;
            s1 += srcstrides;
        }
    }

    if (swap) {
        byte_swap_vector(dst, numitems, itemsize);
    }
}

// private helper to get a default descriptor from a
// possibly NULL dtype, returns NULL on error, which
// can only happen if NPY_DT_CALL_default_descr errors.

static PyArray_Descr *
_infer_descr_from_dtype(PyArray_DTypeMeta *dtype) {
    if (dtype == NULL) {
        return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    if (dtype->singleton != NULL) {
        Py_INCREF(dtype->singleton);
        return dtype->singleton;
    }
    return NPY_DT_CALL_default_descr(dtype);
}

/*
 * Recursive helper to assign using a coercion cache. This function
 * must consume the cache depth first, just as the cache was originally
 * produced.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache_Recursive(
        PyArrayObject *self, const int ndim, coercion_cache_obj **cache)
{
    /* Consume first cache element by extracting information and freeing it */
    PyObject *obj = (*cache)->arr_or_sequence;
    Py_INCREF(obj);
    npy_bool sequence = (*cache)->sequence;
    int depth = (*cache)->depth;
    *cache = npy_unlink_coercion_cache(*cache);

    /* The element is either a sequence, or an array */
    if (!sequence) {
        /* Straight forward array assignment */
        assert(PyArray_Check(obj));
        if (PyArray_CopyInto(self, (PyArrayObject *)obj) < 0) {
            goto fail;
        }
    }
    else {
        assert(depth != ndim);
        npy_intp length = PySequence_Length(obj);
        if (length != PyArray_DIMS(self)[0]) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Inconsistent object during array creation? "
                    "Content of sequences changed (length inconsistent).");
            goto fail;
        }

        for (npy_intp i = 0; i < length; i++) {
            PyObject *value = PySequence_Fast_GET_ITEM(obj, i);

            if (ndim == depth + 1) {
                /*
                 * Straight forward assignment of elements.  Note that it is
                 * possible for such an element to be a 0-D array or array-like.
                 * `PyArray_Pack` supports arrays as well as we want: We
                 * support exact NumPy arrays, but at this point ignore others.
                 * (Please see the `PyArray_Pack` function comment if this
                 * rightly confuses you.)
                 */
                char *item;
                item = (PyArray_BYTES(self) + i * PyArray_STRIDES(self)[0]);
                if (PyArray_Pack(PyArray_DESCR(self), item, value) < 0) {
                    goto fail;
                }
                /* If this was an array(-like) we still need to unlike int: */
                if (*cache != NULL && (*cache)->converted_obj == value) {
                    *cache = npy_unlink_coercion_cache(*cache);
                }
            }
            else {
                PyArrayObject *view;
                view = (PyArrayObject *)array_item_asarray(self, i);
                if (view == NULL) {
                    goto fail;
                }
                if (PyArray_AssignFromCache_Recursive(view, ndim, cache) < 0) {
                    Py_DECREF(view);
                    goto fail;
                }
                Py_DECREF(view);
            }
        }
    }
    Py_DECREF(obj);
    return 0;

  fail:
    Py_DECREF(obj);
    return -1;
}


/**
 * Fills an item based on a coercion cache object. It consumes the cache
 * object while doing so.
 *
 * @param self Array to fill.
 * @param cache coercion_cache_object, will be consumed. The cache must not
 *        contain a single array (must start with a sequence). The array case
 *        should be handled by `PyArray_FromArray()` before.
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache(PyArrayObject *self, coercion_cache_obj *cache) {
    int ndim = PyArray_NDIM(self);
    /*
     * Do not support ndim == 0 now with an array in the cache.
     * The ndim == 0 is special because np.array(np.array(0), dtype=object)
     * should unpack the inner array.
     * Since the single-array case is special, it is handled previously
     * in either case.
     */
    assert(cache->sequence);
    assert(ndim != 0);  /* guaranteed if cache contains a sequence */

    if (PyArray_AssignFromCache_Recursive(self, ndim, &cache) < 0) {
        /* free the remaining cache. */
        npy_free_coercion_cache(cache);
        return -1;
    }

    /*
     * Sanity check, this is the initial call, and when it returns, the
     * cache has to be fully consumed, otherwise something is wrong.
     * NOTE: May be nicer to put into a recursion helper.
     */
    if (cache != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Inconsistent object during array creation? "
                "Content of sequences changed (cache not consumed).");
        npy_free_coercion_cache(cache);
        return -1;
    }
    return 0;
}


static void
raise_memory_error(int nd, npy_intp const *dims, PyArray_Descr *descr)
{
    PyObject *shape = PyArray_IntTupleFromIntp(nd, dims);
    if (shape == NULL) {
        goto fail;
    }

    /* produce an error object */
    PyObject *exc_value = PyTuple_Pack(2, shape, (PyObject *)descr);
    Py_DECREF(shape);
    if (exc_value == NULL){
        goto fail;
    }
    PyErr_SetObject(npy_static_pydata._ArrayMemoryError, exc_value);
    Py_DECREF(exc_value);
    return;

fail:
    /* we couldn't raise the formatted exception for some reason */
    PyErr_WriteUnraisable(NULL);
    PyErr_NoMemory();
}

/*
 * Generic new array creation routine.
 * Internal variant with calloc argument for PyArray_Zeros.
 *
 * steals a reference to descr. On failure or PyDataType_SUBARRAY(descr), descr will
 * be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, _NPY_CREATION_FLAGS cflags)
{
    PyArrayObject_fields *fa;
    npy_intp nbytes;

    if (descr == NULL) {
        return NULL;
    }
    if (nd > NPY_MAXDIMS || nd < 0) {
        PyErr_Format(PyExc_ValueError,
                "number of dimensions must be within [0, %d]", NPY_MAXDIMS);
        Py_DECREF(descr);
        return NULL;
    }

    /* finalize the descriptor if the DType defines a finalization function */
    PyArrayDTypeMeta_FinalizeDescriptor *finalize =
            NPY_DT_SLOTS(NPY_DTYPE(descr))->finalize_descr;
    if (finalize != NULL && data == NULL) {
        Py_SETREF(descr, finalize(descr));
        if (descr == NULL) {
            return NULL;
        }
    }

    nbytes = descr->elsize;
    /*
     * Unless explicitly asked not to, we do replace dtypes in some cases.
     * This mainly means that we never create arrays with a subarray dtype
     * (unless for internal use when requested).  And neither do we create
     * S0/U0 arrays in most cases (unless data == NULL so this is probably
     * a view where growing the dtype would be presumable wrong).
     */
    if (!(cflags & _NPY_ARRAY_ENSURE_DTYPE_IDENTITY)) {
        if (PyDataType_SUBARRAY(descr)) {
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
            ret = PyArray_NewFromDescr_int(
                    subtype, descr,
                    nd, newdims, newstrides, data,
                    flags, obj, base, cflags);
            return ret;
        }

        /* Check datatype element size */
        if (PyDataType_ISUNSIZED(descr)) {
            if (!PyDataType_ISFLEXIBLE(descr) &&
                NPY_DT_is_legacy(NPY_DTYPE(descr))) {
                PyErr_SetString(PyExc_TypeError, "Empty data-type");
                Py_DECREF(descr);
                return NULL;
            }
            else if (PyDataType_ISSTRING(descr)
                        && !(cflags & _NPY_ARRAY_ALLOW_EMPTY_STRING)
                        && data == NULL) {
                PyArray_DESCR_REPLACE(descr);
                if (descr == NULL) {
                    return NULL;
                }
                if (descr->type_num == NPY_STRING) {
                    nbytes = descr->elsize = 1;
                }
                else {
                    nbytes = descr->elsize = sizeof(npy_ucs4);
                }
            }
        }
    }

    fa = (PyArrayObject_fields *) subtype->tp_alloc(subtype, 0);
    if (fa == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    fa->_buffer_info = NULL;
    fa->nd = nd;
    fa->dimensions = NULL;
    fa->data = NULL;
    fa->mem_handler = NULL;

    if (data == NULL) {
        fa->flags = NPY_ARRAY_DEFAULT;
        if (flags) {
            fa->flags |= NPY_ARRAY_F_CONTIGUOUS;
            if (nd > 1) {
                fa->flags &= ~NPY_ARRAY_C_CONTIGUOUS;
            }
            flags = NPY_ARRAY_F_CONTIGUOUS;
        }
    }
    else {
        fa->flags = (flags & ~NPY_ARRAY_WRITEBACKIFCOPY);
    }
    fa->descr = descr;
    fa->base = (PyObject *)NULL;
    fa->weakreflist = (PyObject *)NULL;

    /* needed for zero-filling logic below, defined and initialized up here
       so cleanup logic can go in the fail block */
    NPY_traverse_info fill_zero_info;
    NPY_traverse_info_init(&fill_zero_info);

    if (nd > 0) {
        fa->dimensions = npy_alloc_cache_dim(2 * nd);
        if (fa->dimensions == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        fa->strides = fa->dimensions + nd;

        /*
         * Copy dimensions, check them, and find total array size `nbytes`
         */
        int is_zero = 0;
        for (int i = 0; i < nd; i++) {
            fa->dimensions[i] = dims[i];

            if (fa->dimensions[i] == 0) {
                /*
                 * Continue calculating the max size "as if" this were 1
                 * to get the proper overflow error
                 */
                is_zero = 1;
                continue;
            }

            if (fa->dimensions[i] < 0) {
                PyErr_SetString(PyExc_ValueError,
                        "negative dimensions are not allowed");
                goto fail;
            }

            /*
             * Care needs to be taken to avoid integer overflow when multiplying
             * the dimensions together to get the total size of the array.
             */
            if (npy_mul_sizes_with_overflow(&nbytes, nbytes, fa->dimensions[i])) {
                PyErr_SetString(PyExc_ValueError,
                        "array is too big; `arr.size * arr.dtype.itemsize` "
                        "is larger than the maximum possible size.");
                goto fail;
            }
        }
        if (is_zero) {
            nbytes = 0;
        }

        /* Fill the strides (or copy them if they were passed in) */
        if (strides == NULL) {
            /* fill the strides and set the contiguity flags */
            _array_fill_strides(fa->strides, dims, nd, descr->elsize,
                                flags, &(fa->flags));
        }
        else {
            /* User to provided strides (user is responsible for correctness) */
            for (int i = 0; i < nd; i++) {
                fa->strides[i] = strides[i];
            }
            /* Since the strides were passed in must update contiguity */
            PyArray_UpdateFlags((PyArrayObject *)fa,
                    NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    else {
        fa->dimensions = NULL;
        fa->strides = NULL;
        fa->flags |= NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS;
    }


    if (data == NULL) {
        /* This closely follows PyArray_ZeroContiguousBuffer. We can't use
         *  that because here we need to allocate after checking if there is
         *  custom zeroing logic and that function accepts an already-allocated
         *  array
         */

        /* float errors do not matter and we do not release GIL */
        NPY_ARRAYMETHOD_FLAGS zero_flags;
        PyArrayMethod_GetTraverseLoop *get_fill_zero_loop =
            NPY_DT_SLOTS(NPY_DTYPE(descr))->get_fill_zero_loop;
        if (get_fill_zero_loop != NULL) {
            if (get_fill_zero_loop(
                    NULL, descr, 1, descr->elsize, &(fill_zero_info.func),
                    &(fill_zero_info.auxdata), &zero_flags) < 0) {
                goto fail;
            }
        }

        /*
         * We always want a zero-filled array allocated with calloc if
         * NPY_NEEDS_INIT is set on the dtype, for safety.  We also want a
         * zero-filled array if zeroed is set and the zero-filling loop isn't
         * defined, for better performance.
         *
         * If the zero-filling loop is defined and zeroed is set, allocate
         * with malloc and let the zero-filling loop fill the array buffer
         * with valid zero values for the dtype.
         */
        int use_calloc = (
                PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT) ||
                ((cflags & _NPY_ARRAY_ZEROED) && (fill_zero_info.func == NULL)));

        /* Store the handler in case the default is modified */
        fa->mem_handler = PyDataMem_GetHandler();
        if (fa->mem_handler == NULL) {
            goto fail;
        }
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         */
        if (nbytes == 0) {
            nbytes = 1;
            /* Make sure all the strides are 0 */
            for (int i = 0; i < nd; i++) {
                fa->strides[i] = 0;
            }
        }

        if (use_calloc) {
            data = PyDataMem_UserNEW_ZEROED(nbytes, 1, fa->mem_handler);
        }
        else {
            data = PyDataMem_UserNEW(nbytes, fa->mem_handler);
        }
        if (data == NULL) {
            raise_memory_error(fa->nd, fa->dimensions, descr);
            goto fail;
        }

        /*
         * If the array needs special dtype-specific zero-filling logic, do that
         */
        if (NPY_UNLIKELY((cflags & _NPY_ARRAY_ZEROED)
                         && (fill_zero_info.func != NULL))) {
            npy_intp size = PyArray_MultiplyList(fa->dimensions, fa->nd);
            if (fill_zero_info.func(
                    NULL, descr, data, size, descr->elsize,
                    fill_zero_info.auxdata) < 0) {
                goto fail;
            }
        }

        fa->flags |= NPY_ARRAY_OWNDATA;
    }
    else {
        /* The handlers should never be called in this case */
        fa->mem_handler = NULL;
        /*
         * If data is passed in, this object won't own it.
         */
        fa->flags &= ~NPY_ARRAY_OWNDATA;
    }
    fa->data = data;

    /*
     * Always update the aligned flag.  Not owned data or input strides may
     * not be aligned. Also on some platforms (debian sparc) malloc does not
     * provide enough alignment for long double types.
     */
    PyArray_UpdateFlags((PyArrayObject *)fa, NPY_ARRAY_ALIGNED);

    /* Set the base object. It's important to do it here so that
     * __array_finalize__ below receives it
     */
    if (base != NULL) {
        Py_INCREF(base);
        if (PyArray_SetBaseObject((PyArrayObject *)fa, base) < 0) {
            goto fail;
        }
    }

    /*
     * call the __array_finalize__ method if a subtype was requested.
     * If obj is NULL use Py_None for the Python callback.
     * For speed, we skip if __array_finalize__ is inherited from ndarray
     * (since that function does nothing), or, for backward compatibility,
     * if it is None.
     */
    if (subtype != &PyArray_Type) {
        PyObject *res, *func;
        func = PyObject_GetAttr((PyObject *)subtype, npy_interned_str.array_finalize);
        if (func == NULL) {
            goto fail;
        }
        else if (func == npy_static_pydata.ndarray_array_finalize) {
            Py_DECREF(func);
        }
        else {
            if (PyCapsule_CheckExact(func)) {
                /* A C-function is stored here */
                PyArray_FinalizeFunc *cfunc;
                cfunc = PyCapsule_GetPointer(func, NULL);
                Py_DECREF(func);
                if (cfunc == NULL) {
                    goto fail;
                }
                if (cfunc((PyArrayObject *)fa, obj) < 0) {
                    goto fail;
                }
            }
            else {
                if (obj == NULL) {
                    obj = Py_None;
                }
                res = PyObject_CallFunctionObjArgs(func, (PyObject *)fa, obj, NULL);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
    }
    NPY_traverse_info_xfree(&fill_zero_info);
    return (PyObject *)fa;

 fail:
    NPY_traverse_info_xfree(&fill_zero_info);
    Py_XDECREF(fa->mem_handler);
    Py_DECREF(fa);
    return NULL;
}


/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr. On failure or when PyDataType_SUBARRAY(dtype) is
 * true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj)
{
    if (subtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "subtype is NULL in PyArray_NewFromDescr");
        return NULL;
    }

    if (descr == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "descr is NULL in PyArray_NewFromDescr");
        return NULL;
    }

    return PyArray_NewFromDescrAndBase(
            subtype, descr,
            nd, dims, strides, data,
            flags, obj, NULL);
}

/*
 * Sets the base object using PyArray_SetBaseObject
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base)
{
    return PyArray_NewFromDescr_int(subtype, descr, nd,
                                    dims, strides, data,
                                    flags, obj, base, 0);
}

/*
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order, data type and shape changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * descr     - If not NULL, overrides the data type of the result.
 * dtype     - If not NULL and if descr is NULL, overrides the data type
               of the result, so long as dtype is non-parameteric
 * ndim      - If not -1, overrides the shape of the result.
 * dims      - If ndim is not -1, overrides the shape of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * PyDataType_SUBARRAY(dtype) is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(PyArrayObject *prototype, NPY_ORDER order,
                              PyArray_Descr *descr, PyArray_DTypeMeta *dtype, int ndim,
                              npy_intp const *dims, int subok)
{
    PyObject *ret = NULL;

    if (ndim == -1) {
        ndim = PyArray_NDIM(prototype);
        dims = PyArray_DIMS(prototype);
    }
    else if (order == NPY_KEEPORDER && (ndim != PyArray_NDIM(prototype))) {
        order = NPY_CORDER;
    }

    if (descr == NULL && dtype == NULL) {
        /* If no override data type, use the one from the prototype */
        descr = PyArray_DESCR(prototype);
        Py_INCREF(descr);
    }
    else if (descr == NULL) {
        descr = _infer_descr_from_dtype(dtype);
        if (descr == NULL) {
            return NULL;
        }
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
                                        descr,
                                        ndim,
                                        dims,
                                        NULL,
                                        NULL,
                                        order,
                                        subok ? (PyObject *)prototype : NULL);
    }
    /* KEEPORDER needs some analysis of the strides */
    else {
        npy_intp strides[NPY_MAXDIMS], stride;
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        int idim;

        PyArray_CreateSortedStridePerm(ndim,
                                        PyArray_STRIDES(prototype),
                                        strideperm);

        /* Build the new strides */
        stride = descr->elsize;
        if (stride == 0 && PyDataType_ISSTRING(descr)) {
            /* Special case for dtype=str or dtype=bytes. */
            if (descr->type_num == NPY_STRING) {
                /* dtype is bytes */
                stride = 1;
            }
            else {
                /* dtype is str (type_num is NPY_UNICODE) */
                stride = 4;
            }
        }
        for (idim = ndim-1; idim >= 0; --idim) {
            npy_intp i_perm = strideperm[idim].perm;
            strides[i_perm] = stride;
            stride *= dims[i_perm];
        }

        /* Finally, allocate the array */
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        descr,
                                        ndim,
                                        dims,
                                        strides,
                                        NULL,
                                        0,
                                        subok ? (PyObject *)prototype : NULL);
    }
    if (ret == NULL) {
        return NULL;
    }

    /* Logic shared by `empty`, `empty_like`, and `ndarray.__new__` */
    if (PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)ret))) {
        if (PyArray_SetObjectsToNone((PyArrayObject *)ret) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    return ret;
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
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * PyDataType_SUBARRAY(dtype) is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArray(PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok)
{
    if (prototype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "prototype is NULL in PyArray_NewLikeArray");
        return NULL;
    }
    return PyArray_NewLikeArrayWithShape(prototype, order, dtype, NULL, -1, NULL, subok);
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *subtype, int nd, npy_intp const *dims, int type_num,
        npy_intp const *strides, void *data, int itemsize, int flags,
        PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    if (subtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "subtype is NULL in PyArray_New");
        return NULL;
    }

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (PyDataType_ISUNSIZED(descr)) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        if (descr == NULL) {
            return NULL;
        }
        descr->elsize = itemsize;
    }
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}


NPY_NO_EXPORT PyArray_Descr *
_dtype_from_buffer_3118(PyObject *memoryview)
{
    PyArray_Descr *descr;
    Py_buffer *view = PyMemoryView_GET_BUFFER(memoryview);
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            return NULL;
        }
    }
    else {
        /* If no format is specified, just assume a byte array
         * TODO: void would make more sense here, as it wouldn't null
         *       terminate.
         */
        descr = PyArray_DescrNewFromType(NPY_STRING);
        if (descr == NULL) {
            return NULL;
        }
        descr->elsize = view->itemsize;
    }
    return descr;
}


NPY_NO_EXPORT PyObject *
_array_from_buffer_3118(PyObject *memoryview)
{
    /* PEP 3118 */
    Py_buffer *view;
    PyArray_Descr *descr = NULL;
    PyObject *r = NULL;
    int nd, flags;
    Py_ssize_t d;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    view = PyMemoryView_GET_BUFFER(memoryview);

    if (view->suboffsets != NULL) {
        PyErr_SetString(PyExc_BufferError,
                "NumPy currently does not support importing buffers which "
                "include suboffsets as they are not compatible with the NumPy"
                "memory layout without a copy.  Consider copying the original "
                "before trying to convert it to a NumPy array.");
        return NULL;
    }

    nd = view->ndim;
    descr = _dtype_from_buffer_3118(memoryview);

    if (descr == NULL) {
        return NULL;
    }

    /* Sanity check */
    if (descr->elsize != view->itemsize) {
        /* Ctypes has bugs in its PEP3118 implementation, which we need to
         * work around.
         *
         * bpo-10746
         * bpo-32780
         * bpo-32782
         *
         * Note that even if the above are fixed in main, we have to drop the
         * early patch versions of python to actually make use of the fixes.
         */
        if (!npy_ctypes_check(Py_TYPE(view->obj))) {
            /* This object has no excuse for a broken PEP3118 buffer */
            PyErr_Format(
                    PyExc_RuntimeError,
                   "Item size %zd for PEP 3118 buffer format "
                    "string %s does not match the dtype %c item size %d.",
                    view->itemsize, view->format, descr->type,
                    descr->elsize);
            Py_DECREF(descr);
            return NULL;
        }

        if (PyErr_WarnEx(
                    PyExc_RuntimeWarning,
                    "A builtin ctypes object gave a PEP3118 format "
                    "string that does not match its itemsize, so a "
                    "best-guess will be made of the data type. "
                    "Newer versions of python may behave correctly.", 1) < 0) {
            Py_DECREF(descr);
            return NULL;
        }

        /* Thankfully, np.dtype(ctypes_type) works in most cases.
         * For an array input, this produces a dtype containing all the
         * dimensions, so the array is now 0d.
         */
        nd = 0;
        Py_DECREF(descr);
        descr = (PyArray_Descr *)PyObject_CallFunctionObjArgs(
                (PyObject *)&PyArrayDescr_Type, Py_TYPE(view->obj), NULL);
        if (descr == NULL) {
            return NULL;
        }
        if (descr->elsize != view->len) {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "For the given ctypes object, neither the item size "
                    "computed from the PEP 3118 buffer format nor from "
                    "converting the type to a np.dtype matched the actual "
                    "size. This is a bug both in python and numpy");
            Py_DECREF(descr);
            return NULL;
        }
    }

    if (view->shape != NULL) {
        int k;
        if (nd > NPY_MAXDIMS || nd < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "PEP3118 dimensions do not satisfy 0 <= ndim <= NPY_MAXDIMS");
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
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
                if (view->shape[k] != 0) {
                    d /= view->shape[k];
                }
                strides[k] = d;
            }
        }
    }
    else {
        if (nd == 1) {
            shape[0] = view->len / view->itemsize;
            strides[0] = view->itemsize;
        }
        else if (nd > 1) {
            PyErr_SetString(PyExc_RuntimeError,
                           "ndim computed from the PEP 3118 buffer format "
                           "is greater than 1, but shape is NULL.");
            goto fail;
        }
    }

    flags = NPY_ARRAY_BEHAVED & (view->readonly ? ~NPY_ARRAY_WRITEABLE : ~0);
    r = PyArray_NewFromDescrAndBase(
            &PyArray_Type, descr,
            nd, shape, strides, view->buf,
            flags, NULL, memoryview);
    return r;


fail:
    Py_XDECREF(r);
    Py_XDECREF(descr);
    return NULL;

}


/**
 * Attempts to extract an array from an array-like object.
 *
 * array-like is defined as either
 *
 * * an object implementing the PEP 3118 buffer interface;
 * * an object with __array_struct__ or __array_interface__ attributes;
 * * an object with an __array__ function.
 *
 * @param op The object to convert to an array
 * @param requested_dtype a requested dtype instance, may be NULL; The result
 *                       DType may be used, but is not enforced.
 * @param writeable whether the result must be writeable.
 * @param context Unused parameter, must be NULL (should be removed later).
 * @param copy Specifies the copy behavior.
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy
 *        was made by implementor.
 *
 * @returns The array object, Py_NotImplemented if op is not array-like,
 *          or NULL with an error set. (A new reference to Py_NotImplemented
 *          is returned.)
 */
NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int copy, int *was_copied_by__array__) {
    PyObject* tmp;

    /*
     * If op supports the PEP 3118 buffer interface.
     * We skip bytes and unicode since they are considered scalars. Unicode
     * would fail but bytes would be incorrectly converted to a uint8 array.
     */
    if (PyObject_CheckBuffer(op) && !PyBytes_Check(op) && !PyUnicode_Check(op)) {
        PyObject *memoryview = PyMemoryView_FromObject(op);
        if (memoryview == NULL) {
            /* TODO: Should probably not blanket ignore errors. */
            PyErr_Clear();
        }
        else {
            tmp = _array_from_buffer_3118(memoryview);
            Py_DECREF(memoryview);
            if (tmp == NULL) {
                return NULL;
            }

            if (writeable
                && PyArray_FailUnlessWriteable(
                        (PyArrayObject *)tmp, "PEP 3118 buffer") < 0) {
                Py_DECREF(tmp);
                return NULL;
            }

            return tmp;
        }
    }

    /*
     * If op supports the __array_struct__ or __array_interface__ interface.
     */
    tmp = PyArray_FromStructInterface(op);
    if (tmp == NULL) {
        return NULL;
    }
    if (tmp == Py_NotImplemented) {
        /* Until the return, NotImplemented is always a borrowed reference*/
        tmp = PyArray_FromInterface(op);
        if (tmp == NULL) {
            return NULL;
        }
    }

    if (tmp == Py_NotImplemented) {
        tmp = PyArray_FromArrayAttr_int(
                op, requested_dtype, copy, was_copied_by__array__);
        if (tmp == NULL) {
            return NULL;
        }
    }

    if (tmp != Py_NotImplemented) {
        if (writeable &&
                PyArray_FailUnlessWriteable((PyArrayObject *)tmp,
                        "array interface object") < 0) {
            Py_DECREF(tmp);
            return NULL;
        }
        return tmp;
    }

    /* Until here Py_NotImplemented was borrowed */
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


/*NUMPY_API
 * Does not check for NPY_ARRAY_ENSURECOPY and NPY_ARRAY_NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    npy_dtype_info dt_info = {NULL, NULL};

    int res = PyArray_ExtractDTypeAndDescriptor(
        newtype, &dt_info.descr, &dt_info.dtype);

    Py_XDECREF(newtype);

    if (res < 0) {
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    /*
     * The internal implementation treats 0 as actually wanting a zero-dimensional
     * array, but the API for this function has typically treated it as
     * "anything is fine", so convert here.
     * TODO: should we use another value as a placeholder instead?
     */
    if (max_depth == 0 || max_depth > NPY_MAXDIMS) {
        max_depth = NPY_MAXDIMS;
    }

    int was_scalar;
    PyObject* ret =  PyArray_FromAny_int(
            op, dt_info.descr, dt_info.dtype,
            min_depth, max_depth, flags, context, &was_scalar);

    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return ret;
}

/*
 * Internal version of PyArray_FromAny that accepts a dtypemeta. Borrows
 * references to the descriptor and dtype.
 *
 * The `was_scalar` output returns 1 when the object was a "scalar".
 * This means it was:
 * - Recognized as a scalar by a/the dtype.  This can be DType specific,
 *   for example a tuple may be a scalar, but only for structured dtypes.
 * - Anything not recognized as an instance of a DType's scalar type but also not
 *   convertible to an array.  (no __array__ protocol, etc.)
 *   these must map to `dtype=object` (if a dtype wasn't specified).
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny_int(PyObject *op, PyArray_Descr *in_descr,
                    PyArray_DTypeMeta *in_DType, int min_depth, int max_depth,
                    int flags, PyObject *context, int *was_scalar)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from many different places.
     */
    PyArrayObject *arr = NULL, *ret = NULL;
    PyArray_Descr *dtype = NULL;
    coercion_cache_obj *cache = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    if (context != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    // Default is copy = None
    int copy = -1;
    int was_copied_by__array__ = 0;

    if (flags & NPY_ARRAY_ENSURENOCOPY) {
        copy = 0;
    } else if (flags & NPY_ARRAY_ENSURECOPY) {
        copy = 1;
    }

    Py_BEGIN_CRITICAL_SECTION(op);

    ndim = PyArray_DiscoverDTypeAndShape(
            op, max_depth, dims, &cache, in_DType, in_descr, &dtype,
            copy, &was_copied_by__array__);

    if (ndim < 0) {
        goto cleanup;
    }

    /* If the cache is NULL, then the object is considered a scalar */
    *was_scalar = (cache == NULL);

    if (dtype == NULL) {
        dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    if (min_depth != 0 && ndim < min_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object of too small depth for desired array");
        npy_free_coercion_cache(cache);
        goto cleanup;
    }
    if (ndim > max_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object too deep for desired array");
        npy_free_coercion_cache(cache);
        goto cleanup;
    }

    /* Got the correct parameters, but the cache may already hold the result */
    if (cache != NULL && !(cache->sequence)) {
        /*
         * There is only a single array-like and it was converted, it
         * may still have the incorrect type, but that is handled below.
         */
        assert(cache->converted_obj == op);
        arr = (PyArrayObject *)(cache->arr_or_sequence);
        /* we may need to cast or assert flags (e.g. copy) */
        if (was_copied_by__array__ == 1) {
            flags = flags & ~NPY_ARRAY_ENSURECOPY;
        }
        // PyArray_FromArray steals a reference to the dtype
        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_FromArray(arr, dtype, flags);
        npy_unlink_coercion_cache(cache);
        goto cleanup;
    }
    else if (cache == NULL && PyArray_IsScalar(op, Void) &&
            !(((PyVoidScalarObject *)op)->flags & NPY_ARRAY_OWNDATA) &&
             ((in_descr == NULL) && (in_DType == NULL))) {
        /*
         * Special case, we return a *view* into void scalars, mainly to
         * allow things similar to the "reversed" assignment:
         *    arr[indx]["field"] = val  # instead of arr["field"][indx] = val
         *
         * It is unclear that this is necessary in this particular code path.
         * Note that this path is only activated when the user did _not_
         * provide a dtype (newtype is NULL).
         */
        assert(ndim == 0);
        // PyArray_NewFromDescrAndBase steals a reference to the dtype
        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type, dtype,
                0, NULL, NULL,
                ((PyVoidScalarObject *)op)->obval,
                ((PyVoidScalarObject *)op)->flags,
                NULL, op);
        goto cleanup;
    }
    /*
     * If we got this far, we definitely have to create a copy, since we are
     * converting either from a scalar (cache == NULL) or a (nested) sequence.
     */
    if (flags & NPY_ARRAY_ENSURENOCOPY) {
        PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
        npy_free_coercion_cache(cache);
        goto cleanup;
    }

    if (cache == NULL && in_descr != NULL &&
            PyDataType_ISSIGNED(dtype) &&
            PyArray_IsScalar(op, Generic)) {
        assert(ndim == 0);
        /*
         * This is an (possible) inconsistency where:
         *
         *     np.array(np.float64(np.nan), dtype=np.int64)
         *
         * behaves differently from:
         *
         *     np.array([np.float64(np.nan)], dtype=np.int64)
         *     arr1d_int64[0] = np.float64(np.nan)
         *     np.array(np.array(np.nan), dtype=np.int64)
         *
         * by not raising an error instead of using typical casting.
         * The error is desirable, but to always error seems like a
         * larger change to be considered at some other time and it is
         * undesirable that 0-D arrays behave differently from scalars.
         * This retains the behaviour, largely due to issues in pandas
         * which relied on a try/except (although hopefully that will
         * have a better solution at some point):
         * https://github.com/pandas-dev/pandas/issues/35481
         */
        // PyArray_FromScalar steals a reference to dtype
        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_FromScalar(op, dtype);
        goto cleanup;
    }

    /* There was no array (or array-like) passed in directly. */
    if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
        PyErr_SetString(PyExc_TypeError,
                        "WRITEBACKIFCOPY used for non-array input.");
        npy_free_coercion_cache(cache);
        goto cleanup;
    }

    /* Create a new array and copy the data */
    Py_INCREF(dtype);  /* hold on in case of a subarray that is replaced */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, dtype, ndim, dims, NULL, NULL,
            flags&NPY_ARRAY_F_CONTIGUOUS, NULL);
    if (ret == NULL) {
        npy_free_coercion_cache(cache);
        goto cleanup;
    }
    if (ndim == PyArray_NDIM(ret)) {
        /*
         * Appending of dimensions did not occur, so use the actual dtype
         * below. This is relevant for S0 or U0 which can be replaced with
         * S1 or U1, although that should likely change.
         */
        Py_SETREF(dtype, PyArray_DESCR(ret));
        Py_INCREF(dtype);
    }

    if (cache == NULL) {
        /* This is a single item. Set it directly. */
        assert(ndim == 0);

        if (PyArray_Pack(dtype, PyArray_BYTES(ret), op) < 0) {
            Py_CLEAR(ret);
            goto cleanup;
        }
        goto cleanup;
    }
    assert(ndim != 0);
    assert(op == cache->converted_obj);

    /* Decrease the number of dimensions to the detected ones */
    int out_ndim = PyArray_NDIM(ret);
    PyArray_Descr *out_descr = PyArray_DESCR(ret);
    if (out_ndim != ndim) {
        ((PyArrayObject_fields *)ret)->nd = ndim;
        ((PyArrayObject_fields *)ret)->descr = dtype;
    }

    int succeed = PyArray_AssignFromCache(ret, cache);

    ((PyArrayObject_fields *)ret)->nd = out_ndim;
    ((PyArrayObject_fields *)ret)->descr = out_descr;
    if (succeed < 0) {
        Py_CLEAR(ret);
    }

cleanup:;

    Py_XDECREF(dtype);
    Py_END_CRITICAL_SECTION();
    return (PyObject *)ret;
}

/*
 * flags is any of
 * NPY_ARRAY_C_CONTIGUOUS (formerly CONTIGUOUS),
 * NPY_ARRAY_F_CONTIGUOUS (formerly FORTRAN),
 * NPY_ARRAY_ALIGNED,
 * NPY_ARRAY_WRITEABLE,
 * NPY_ARRAY_NOTSWAPPED,
 * NPY_ARRAY_ENSURECOPY,
 * NPY_ARRAY_WRITEBACKIFCOPY,
 * NPY_ARRAY_FORCECAST,
 * NPY_ARRAY_ENSUREARRAY,
 * NPY_ARRAY_ELEMENTSTRIDES,
 * NPY_ARRAY_ENSURENOCOPY
 *
 * or'd (|) together
 *
 * Any of these flags present means that the returned array should
 * guarantee that aspect of the array.  Otherwise the returned array
 * won't guarantee it -- it will depend on the object as to whether or
 * not it has such features.
 *
 * Note that NPY_ARRAY_ENSURECOPY is enough
 * to guarantee NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_ALIGNED and
 * NPY_ARRAY_WRITEABLE and therefore it is redundant to include
 * those as well.
 *
 * NPY_ARRAY_BEHAVED == NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
 * NPY_ARRAY_CARRAY = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED
 * NPY_ARRAY_FARRAY = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED
 *
 * NPY_ARRAY_F_CONTIGUOUS can be set in the FLAGS to request a FORTRAN array.
 * Fortran arrays are always behaved (aligned,
 * notswapped, and writeable) and not (C) CONTIGUOUS (if > 1d).
 *
 * NPY_ARRAY_WRITEBACKIFCOPY flag sets this flag in the returned
 * array if a copy is made and the base argument points to the (possibly)
 * misbehaved array. Before returning to python, PyArray_ResolveWritebackIfCopy
 * must be called to update the contents of the original array from the copy.
 *
 * NPY_ARRAY_FORCECAST will cause a cast to occur regardless of whether or not
 * it is safe.
 *
 */

/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    npy_dtype_info dt_info = {NULL, NULL};

    int res = PyArray_ExtractDTypeAndDescriptor(
        descr, &dt_info.descr, &dt_info.dtype);

    Py_XDECREF(descr);

    if (res < 0) {
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    /* See comment in PyArray_FromAny for rationale */
    if (max_depth == 0 || max_depth > NPY_MAXDIMS) {
        max_depth = NPY_MAXDIMS;
    }

    PyObject* ret =  PyArray_CheckFromAny_int(
        op, dt_info.descr, dt_info.dtype, min_depth, max_depth, requires,
        context);

    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return ret;
}

/*
 * Internal version of PyArray_CheckFromAny that accepts a dtypemeta. Borrows
 * references to the descriptor and dtype.
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny_int(PyObject *op, PyArray_Descr *in_descr,
                         PyArray_DTypeMeta *in_DType, int min_depth,
                         int max_depth, int requires, PyObject *context)
{
    PyObject *obj;
    Py_XINCREF(in_descr);  /* take ownership as we may replace it */
    if (requires & NPY_ARRAY_NOTSWAPPED) {
        if (!in_descr && PyArray_Check(op)) {
            in_descr = PyArray_DESCR((PyArrayObject *)op);
            Py_INCREF(in_descr);
        }
        if (in_descr) {
            PyArray_DESCR_REPLACE_CANONICAL(in_descr);
            if (in_descr == NULL) {
                return NULL;
            }
        }
    }

    int was_scalar;
    obj = PyArray_FromAny_int(op, in_descr, in_DType, min_depth,
                              max_depth, requires, context, &was_scalar);
    Py_XDECREF(in_descr);
    if (obj == NULL) {
        return NULL;
    }

    if ((requires & NPY_ARRAY_ELEMENTSTRIDES)
            && !PyArray_ElementStrides(obj)) {
        PyObject *ret;
        if (requires & NPY_ARRAY_ENSURENOCOPY) {
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            return NULL;
        }
        ret = PyArray_NewCopy((PyArrayObject *)obj, NPY_ANYORDER);
        Py_DECREF(obj);
        obj = ret;
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
    int copy = 0;
    int arrflags;
    PyArray_Descr *oldtype;
    NPY_CASTING casting = NPY_SAFE_CASTING;

    oldtype = PyArray_DESCR(arr);
    if (newtype == NULL) {
        /*
         * Check if object is of array with Null newtype.
         * If so return it directly instead of checking for casting.
         */
        if (flags == 0) {
            Py_INCREF(arr);
            return (PyObject *)arr;
        }
        newtype = oldtype;
        Py_INCREF(oldtype);
    }
    else if (PyDataType_ISUNSIZED(newtype)) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
    }

    if (flags & NPY_ARRAY_SAME_KIND_CASTING) {
        casting = NPY_SAME_KIND_CASTING;
    }

    /* If the casting if forced, use the 'unsafe' casting rule */
    if (flags & NPY_ARRAY_FORCECAST) {
        casting = NPY_UNSAFE_CASTING;
    }

    /* Raise an error if the casting rule isn't followed */
    if (!PyArray_CanCastArrayTo(arr, newtype, casting)) {
        PyErr_Clear();
        npy_set_invalid_cast_error(
                PyArray_DESCR(arr), newtype, casting, PyArray_NDIM(arr) == 0);
        Py_DECREF(newtype);
        return NULL;
    }

    arrflags = PyArray_FLAGS(arr);


    copy = /* If a guaranteed copy was requested */
           (flags & NPY_ARRAY_ENSURECOPY) ||
           /* If C contiguous was requested, and arr is not */
           ((flags & NPY_ARRAY_C_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_C_CONTIGUOUS))) ||
           /* If an aligned array was requested, and arr is not */
           ((flags & NPY_ARRAY_ALIGNED) &&
                   (!(arrflags & NPY_ARRAY_ALIGNED))) ||
           /* If a Fortran contiguous array was requested, and arr is not */
           ((flags & NPY_ARRAY_F_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_F_CONTIGUOUS))) ||
           /* If a writeable array was requested, and arr is not */
           ((flags & NPY_ARRAY_WRITEABLE) &&
                   (!(arrflags & NPY_ARRAY_WRITEABLE)));

    if (!copy) {
        npy_intp view_offset;
        npy_intp is_safe = PyArray_SafeCast(oldtype, newtype, &view_offset, NPY_NO_CASTING, 1);
        copy = !(is_safe && (view_offset != NPY_MIN_INTP));
    }

    if (copy) {
        if (flags & NPY_ARRAY_ENSURENOCOPY) {
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            Py_DECREF(newtype);
            return NULL;
        }

        NPY_ORDER order = NPY_KEEPORDER;
        int subok = 1;

        /* Set the order for the copy being made based on the flags */
        if (flags & NPY_ARRAY_F_CONTIGUOUS) {
            order = NPY_FORTRANORDER;
        }
        else if (flags & NPY_ARRAY_C_CONTIGUOUS) {
            order = NPY_CORDER;
        }

        if ((flags & NPY_ARRAY_ENSUREARRAY)) {
            subok = 0;
        }
        Py_INCREF(newtype);
        ret = (PyArrayObject *)PyArray_NewLikeArray(arr, order,
                                                    newtype, subok);
        if (ret == NULL) {
            Py_DECREF(newtype);
            return NULL;
        }

        int actual_ndim = PyArray_NDIM(ret);
        PyArray_Descr *actual_dtype = PyArray_DESCR(ret);
        if (actual_ndim != PyArray_NDIM(arr)) {
            ((PyArrayObject_fields *)ret)->nd = PyArray_NDIM(arr);
            ((PyArrayObject_fields *)ret)->descr = newtype;
        }

        int success = PyArray_CopyInto(ret, arr);

        Py_DECREF(newtype);
        ((PyArrayObject_fields *)ret)->nd = actual_ndim;
        ((PyArrayObject_fields *)ret)->descr = actual_dtype;

        if (success < 0) {
            Py_DECREF(ret);
            return NULL;
        }


        if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            Py_INCREF(arr);
            if (PyArray_SetWritebackIfCopyBase(ret, arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
        }
    }
    /*
     * If no copy then take an appropriate view if necessary, or
     * just return a reference to ret itself.
     */
    else {
        int needview = ((flags & NPY_ARRAY_ENSUREARRAY) &&
                        !PyArray_CheckExact(arr));

        Py_DECREF(newtype);
        if (needview) {
            PyTypeObject *subtype = NULL;

            if (flags & NPY_ARRAY_ENSUREARRAY) {
                subtype = &PyArray_Type;
            }
            ret = (PyArrayObject *)PyArray_View(arr, NULL, subtype);
            if (ret == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(arr);
            ret = arr;
        }
    }

    return (PyObject *)ret;
}

/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    PyArray_Descr *thetype = NULL;
    PyArrayInterface *inter;
    PyObject *attr;
    char endian = NPY_NATBYTE;

    if (PyArray_LookupSpecial_OnInstance(
            input, npy_interned_str.array_struct, &attr) < 0) {
        return NULL;
    }
    else if (attr == NULL) {
        return Py_NotImplemented;
    }
    if (!PyCapsule_CheckExact(attr)) {
        if (PyType_Check(input) && PyObject_HasAttrString(attr, "__get__")) {
            /*
             * If the input is a class `attr` should be a property-like object.
             * This cannot be interpreted as an array, but is a valid.
             * (Needed due to the lookup being on the instance rather than type)
             */
            Py_DECREF(attr);
            return Py_NotImplemented;
        }
        goto fail;
    }
    inter = PyCapsule_GetPointer(attr, NULL);
    if (inter == NULL) {
        goto fail;
    }
    if (inter->two != 2) {
        goto fail;
    }
    if ((inter->flags & NPY_ARRAY_NOTSWAPPED) != NPY_ARRAY_NOTSWAPPED) {
        endian = NPY_OPPBYTE;
        inter->flags &= ~NPY_ARRAY_NOTSWAPPED;
    }

    if (inter->flags & NPY_ARR_HAS_DESCR) {
        if (PyArray_DescrConverter(inter->descr, &thetype) == NPY_FAIL) {
            thetype = NULL;
            PyErr_Clear();
        }
    }

    if (thetype == NULL) {
        PyObject *type_str = PyUnicode_FromFormat(
            "%c%c%d", endian, inter->typekind, inter->itemsize);
        if (type_str == NULL) {
            Py_DECREF(attr);
            return NULL;
        }
        int ok = PyArray_DescrConverter(type_str, &thetype);
        Py_DECREF(type_str);
        if (ok != NPY_SUCCEED) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    /* a tuple to hold references */
    PyObject *refs = PyTuple_New(2);
    if (!refs) {
        Py_DECREF(attr);
        return NULL;
    }

    /* add a reference to the object sharing the data */
    Py_INCREF(input);
    PyTuple_SET_ITEM(refs, 0, input);

    /* take a reference to the PyCapsule containing the PyArrayInterface
     * structure. When the PyCapsule reference is released the PyCapsule
     * destructor will free any resources that need to persist while numpy has
     * access to the data. */
    PyTuple_SET_ITEM(refs, 1,  attr);

    /* create the numpy array, this call adds a reference to refs */
    PyObject *ret = PyArray_NewFromDescrAndBase(
            &PyArray_Type, thetype,
            inter->nd, inter->shape, inter->strides, inter->data,
            inter->flags, NULL, refs);

    Py_DECREF(refs);

    return ret;

 fail:
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    Py_DECREF(attr);
    return NULL;
}

/*
 * Checks if the object in descr is the default 'descr' member for the
 * __array_interface__ dictionary with 'typestr' member typestr.
 */
NPY_NO_EXPORT int
_is_default_descr(PyObject *descr, PyObject *typestr) {
    if (!PyList_Check(descr) || PyList_GET_SIZE(descr) != 1) {
        return 0;
    }
    PyObject *tuple = PyList_GET_ITEM(descr, 0); // noqa: borrowed-ref - manual fix needed
    if (!(PyTuple_Check(tuple) && PyTuple_GET_SIZE(tuple) == 2)) {
        return 0;
    }
    PyObject *name = PyTuple_GET_ITEM(tuple, 0);
    if (!(PyUnicode_Check(name) && PyUnicode_GetLength(name) == 0)) {
        return 0;
    }
    PyObject *typestr2 = PyTuple_GET_ITEM(tuple, 1);
    return PyObject_RichCompareBool(typestr, typestr2, Py_EQ);
}


/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *origin)
{
    PyObject *iface = NULL;
    PyObject *attr = NULL;
    PyObject *base = NULL;
    PyArrayObject *ret;
    PyArray_Descr *dtype = NULL;
    char *data = NULL;
    Py_buffer view;
    Py_ssize_t i, n;
    npy_intp dims[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    int dataflags = NPY_ARRAY_BEHAVED;
    int use_scalar_assign = 0;

    if (PyArray_LookupSpecial_OnInstance(
            origin, npy_interned_str.array_interface, &iface) < 0) {
        return NULL;
    }
    else if (iface == NULL) {
        return Py_NotImplemented;
    }
    if (!PyDict_Check(iface)) {
        if (PyType_Check(origin) && PyObject_HasAttrString(iface, "__get__")) {
            /*
             * If the input is a class `iface` should be a property-like object.
             * This cannot be interpreted as an array, but is a valid.
             * (Needed due to the lookup being on the instance rather than type)
             */
            Py_DECREF(iface);
            return Py_NotImplemented;
        }

        Py_DECREF(iface);
        PyErr_SetString(PyExc_ValueError,
                "Invalid __array_interface__ value, must be a dict");
        return NULL;
    }

    /* Get type string from interface specification */
    int result = PyDict_GetItemStringRef(iface, "typestr", &attr);
    if (result <= 0) {
        Py_DECREF(iface);
        if (result == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ typestr");
        }
        return NULL;
    }

    /* allow bytes for backwards compatibility */
    if (!PyBytes_Check(attr) && !PyUnicode_Check(attr)) {
        PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ typestr must be a string");
        goto fail;
    }

    /* Get dtype from type string */
    if (PyArray_DescrConverter(attr, &dtype) != NPY_SUCCEED) {
        goto fail;
    }

    /*
     * If the dtype is NPY_VOID, see if there is extra information in
     * the 'descr' attribute.
     */
    if (dtype->type_num == NPY_VOID) {
        PyObject *descr = NULL;
        result = PyDict_GetItemStringRef(iface, "descr", &descr);
        if (result == -1) {
            goto fail;
        }
        PyArray_Descr *new_dtype = NULL;
        if (result == 1) {
            int is_default = _is_default_descr(descr, attr);
            if (is_default < 0) {
                Py_DECREF(descr);
                goto fail;
            }
            if (!is_default) {
                if (PyArray_DescrConverter2(descr, &new_dtype) != NPY_SUCCEED) {
                    Py_DECREF(descr);
                    goto fail;
                }
                if (new_dtype != NULL) {
                    Py_SETREF(dtype, new_dtype);
                }
            }
            Py_DECREF(descr);
        }
    }
    Py_CLEAR(attr);

    /* Get shape tuple from interface specification */
    result = PyDict_GetItemStringRef(iface, "shape", &attr);
    if (result < 0) {
        return NULL;
    }
    if (result == 0) {
        /* Shape must be specified when 'data' is specified */
        int result = PyDict_ContainsString(iface, "data");
        if (result < 0) {
            return NULL;
        }
        else if (result == 1) {
            Py_DECREF(iface);
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ shape");
            return NULL;
        }
        /* Assume shape as scalar otherwise */
        else {
            /* NOTE: pointers to data and base should be NULL */
            n = dims[0] = 0;
        }
    }
    /* Make sure 'shape' is a tuple */
    else if (!PyTuple_Check(attr)) {
        PyErr_SetString(PyExc_TypeError,
                "shape must be a tuple");
        goto fail;
    }
    /* Get dimensions from shape tuple */
    else {
        n = PyTuple_GET_SIZE(attr);
        if (n > NPY_MAXDIMS) {
            PyErr_Format(PyExc_ValueError,
                         "number of dimensions must be within [0, %d], got %d",
                         NPY_MAXDIMS, n);
            goto fail;
        }
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            dims[i] = PyArray_PyIntAsIntp(tmp);
            if (error_converting(dims[i])) {
                goto fail;
            }
        }
    }
    Py_CLEAR(attr);

    /* Get data buffer from interface specification */
    result = PyDict_GetItemStringRef(iface, "data", &attr);
    if (result == -1){
        return NULL;
    }

    /* Case for data access through pointer */
    if (attr == NULL) {
        use_scalar_assign = 1;
    }
    else if (PyTuple_Check(attr)) {
        PyObject *dataptr;
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ data must be a 2-tuple with "
                    "(data pointer integer, read-only flag)");
            goto fail;
        }
        dataptr = PyTuple_GET_ITEM(attr, 0);
        if (PyLong_Check(dataptr)) {
            data = PyLong_AsVoidPtr(dataptr);
            if (data == NULL && PyErr_Occurred()) {
                goto fail;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "first element of __array_interface__ data tuple "
                    "must be an integer.");
            goto fail;
        }
        int istrue = PyObject_IsTrue(PyTuple_GET_ITEM(attr,1));
        if (istrue == -1) {
            goto fail;
        }
        if (istrue) {
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        base = origin;
    }

    /* Case for data access through buffer */
    else {
        if (attr != Py_None) {
            base = attr;
        }
        else {
            base = origin;
        }
        if (PyObject_GetBuffer(base, &view,
                    PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
            PyErr_Clear();
            if (PyObject_GetBuffer(base, &view,
                        PyBUF_SIMPLE) < 0) {
                goto fail;
            }
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        data = (char *)view.buf;
        /*
         * Both of the deprecated functions PyObject_AsWriteBuffer and
         * PyObject_AsReadBuffer that this code replaces release the buffer. It is
         * up to the object that supplies the buffer to guarantee that the buffer
         * sticks around after the release.
         */
        PyBuffer_Release(&view);

        /* Get offset number from interface specification */
        PyObject *offset = NULL;
        result = PyDict_GetItemStringRef(iface, "offset", &offset);
        if (result == -1) {
            goto fail;
        }
        else if (result == 1) {
            npy_longlong num = PyLong_AsLongLong(offset);
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                        "__array_interface__ offset must be an integer");
                Py_DECREF(offset);
                goto fail;
            }
            data += num;
            Py_DECREF(offset);
        }
    }
    Py_CLEAR(attr);

    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            n, dims, NULL, data,
            dataflags, NULL, base);
    /*
     * Ref to dtype was stolen by PyArray_NewFromDescrAndBase
     * Prevent DECREFing dtype in fail codepath by setting to NULL
     */
    dtype = NULL;
    if (ret == NULL) {
        goto fail;
    }
    if (use_scalar_assign) {
        /* 
         * NOTE(seberg): I honestly doubt anyone is using this scalar path and we
         * could probably just deprecate (or just remove it in a 3.0 version).
         */
        if (PyArray_SIZE(ret) > 1) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot coerce scalar to array with size > 1");
            Py_DECREF(ret);
            goto fail;
        }
        if (PyArray_Pack(PyArray_DESCR(ret), PyArray_DATA(ret), origin) < 0) {
            Py_DECREF(ret);
            goto fail;
        }
    }
    else if (data == NULL && PyArray_NBYTES(ret) != 0) {
        /* Caller should ensure this, but <2.4 used the above scalar coerction path */
        PyErr_SetString(PyExc_ValueError,
                "data is NULL but array contains data, in older versions of NumPy "
                "this may have used the scalar path.  To get the scalar path "
                "you must leave the data field undefined.");
        Py_DECREF(ret);
        goto fail;
    }

    result = PyDict_GetItemStringRef(iface, "strides", &attr);
    if (result == -1){
        return NULL;
    }
    if (result == 1 && attr != Py_None) {
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                    "strides must be a tuple");
            Py_DECREF(ret);
            goto fail;
        }
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                    "mismatch in length of strides and shape");
            Py_DECREF(ret);
            goto fail;
        }
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(tmp);
            if (error_converting(strides[i])) {
                Py_DECREF(ret);
                goto fail;
            }
        }
        if (n) {
            memcpy(PyArray_STRIDES(ret), strides, n*sizeof(npy_intp));
        }
        Py_DECREF(attr);
    }
    PyArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);
    Py_DECREF(iface);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(attr);
    Py_XDECREF(dtype);
    Py_XDECREF(iface);
    return NULL;
}



/*
 * Returns -1 and an error set or 0 with the original error cleared, must
 * be called with an error set.
 */
static inline int
check_or_clear_and_warn_error_if_due_to_copy_kwarg(PyObject *kwnames)
{
    if (kwnames == NULL) {
        return -1;  /* didn't pass kwnames, can't possibly be the reason */
    }
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        return -1;
    }

    /*
     * In most cases, if we fail, we assume the error was unrelated to the
     * copy kwarg and simply restore the original one.
     */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (value == NULL) {
        goto restore_error;
    }

    PyObject *str_value = PyObject_Str(value);
    if (str_value == NULL) {
        goto restore_error;
    }
    int copy_kwarg_unsupported = PyUnicode_Contains(
            str_value, npy_interned_str.array_err_msg_substr);
    Py_DECREF(str_value);
    if (copy_kwarg_unsupported == -1) {
        goto restore_error;
    }
    if (copy_kwarg_unsupported) {
        /*
         * TODO: As of now NumPy 2.0, the this warning is only triggered with
         *       `copy=False` allowing downstream to not notice it.
         */
        Py_DECREF(type);
        Py_DECREF(value);
        Py_XDECREF(traceback);
        if (DEPRECATE("__array__ implementation doesn't accept a copy keyword, "
                      "so passing copy=False failed. __array__ must implement "
                      "'dtype' and 'copy' keyword arguments. "
                      "To learn more, see the migration guide "
                      "https://numpy.org/devdocs/numpy_2_0_migration_guide.html"
                      "#adapting-to-changes-in-the-copy-keyword") < 0) {
            return -1;
        }
        return 0;
    }

  restore_error:
    PyErr_Restore(type, value, traceback);
    return -1;
}


/**
 * Check for an __array__ attribute and call it when it exists.
 *
 *  .. warning:
 *      If returned, `NotImplemented` is borrowed and must not be Decref'd
 *
 * @param op The Python object to convert to an array.
 * @param descr The desired `arr.dtype`, passed into the `__array__` call,
 *        as information but is not checked/enforced!
 * @param copy Specifies the copy behavior
 *        NOTE: For copy == -1 it passes `op.__array__(copy=None)`,
 *              for copy == 0, `op.__array__(copy=False)`, and
 *              for copy == 1, `op.__array__(copy=True).
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy
 *        was made by implementor.
 * @returns NotImplemented if `__array__` is not defined or a NumPy array
 *          (or subclass).  On error, return NULL.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr_int(PyObject *op, PyArray_Descr *descr, int copy,
                          int *was_copied_by__array__)
{
    PyObject *new;
    PyObject *array_meth;

    if (PyArray_LookupSpecial_OnInstance(
                op, npy_interned_str.array, &array_meth) < 0) {
        return NULL;
    }
    else if (array_meth == NULL) {
        return Py_NotImplemented;
    }

    if (PyType_Check(op) && PyObject_HasAttrString(array_meth, "__get__")) {
        /*
         * If the input is a class `array_meth` may be a property-like object.
         * This cannot be interpreted as an array (called), but is a valid.
         * Trying `array_meth.__call__()` on this should not be useful.
         * (Needed due to the lookup being on the instance rather than type)
         */
        Py_DECREF(array_meth);
        return Py_NotImplemented;
    }

    Py_ssize_t nargs = 0;
    PyObject *arguments[2];
    PyObject *kwnames = NULL;

    if (descr != NULL) {
        arguments[0] = (PyObject *)descr;
        nargs++;
    }

    /*
     * Only if the value of `copy` isn't the default one, we try to pass it
     * along; for backwards compatibility we then retry if it fails because the
     * signature of the __array__ method being called does not have `copy`.
     */
    if (copy != -1) {
        kwnames = npy_static_pydata.kwnames_is_copy;
        arguments[nargs] = copy == 1 ? Py_True : Py_False;
    }

    int must_copy_but_copy_kwarg_unimplemented = 0;
    new = PyObject_Vectorcall(array_meth, arguments, nargs, kwnames);
    if (new == NULL) {
        if (check_or_clear_and_warn_error_if_due_to_copy_kwarg(kwnames) < 0) {
            /* Error was not cleared (or a new error set) */
            Py_DECREF(array_meth);
            return NULL;
        }
        if (copy == 0) {
            /* Cannot possibly avoid a copy, so error out. */
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            Py_DECREF(array_meth);
            return NULL;
        }
        /*
         * The error seems to have been due to passing copy.  We try to see
         * more precisely what the message is and may try again.
         */
        must_copy_but_copy_kwarg_unimplemented = 1;
        new = PyObject_Vectorcall(array_meth, arguments, nargs, NULL);
        if (new == NULL) {
            Py_DECREF(array_meth);
            return NULL;
        }
    }

    Py_DECREF(array_meth);

    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not "  \
                        "producing an array");
        Py_DECREF(new);
        return NULL;
    }
    /* TODO: Remove was_copied_by__array__ argument */
    if (was_copied_by__array__ != NULL && copy == 1 &&
        must_copy_but_copy_kwarg_unimplemented == 0) {
        /* We can assume that a copy was made */
        *was_copied_by__array__ = 1;
    }

    return new;
}


/*NUMPY_API
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    if (context != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    return PyArray_FromArrayAttr_int(op, typecode, 0, NULL);
}


/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
{
    PyArray_Descr *dtype;

    dtype = mintype;
    Py_XINCREF(dtype);

    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NULL;
    }

    if (dtype == NULL) {
        return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    else {
        return dtype;
    }
}


/*NUMPY_API
 * This is a quick wrapper around
 * PyArray_FromAny(op, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL)
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
        new = PyArray_FROM_OF(op, NPY_ARRAY_ENSUREARRAY);
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

/*
 * Private implementation of PyArray_CopyAnyInto with an additional order
 * parameter.
 */
NPY_NO_EXPORT int
PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src, NPY_ORDER order)
{
    NpyIter *dst_iter, *src_iter;

    NpyIter_IterNextFunc *dst_iternext, *src_iternext;
    char **dst_dataptr, **src_dataptr;
    npy_intp dst_stride, src_stride;
    npy_intp *dst_countptr, *src_countptr;
    npy_uint32 baseflags;

    npy_intp dst_count, src_count, count;
    npy_intp dst_size, src_size;

    NPY_BEGIN_THREADS_DEF;

    if (PyArray_FailUnlessWriteable(dst, "destination array") < 0) {
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
        PyErr_Format(PyExc_ValueError,
                "cannot copy from array of size %" NPY_INTP_FMT " into an array "
                "of size %" NPY_INTP_FMT, src_size, dst_size);
        return -1;
    }

    /* Zero-sized arrays require nothing be done */
    if (dst_size == 0) {
        return 0;
    }

    baseflags = NPY_ITER_EXTERNAL_LOOP |
                NPY_ITER_DONT_NEGATE_STRIDES |
                NPY_ITER_REFS_OK;

    /*
     * This copy is based on matching C-order traversals of src and dst.
     * By using two iterators, we can find maximal sub-chunks that
     * can be processed at once.
     */
    dst_iter = NpyIter_New(dst, NPY_ITER_WRITEONLY | baseflags,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (dst_iter == NULL) {
        return -1;
    }
    src_iter = NpyIter_New(src, NPY_ITER_READONLY | baseflags,
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
    /* The inner stride is also the fixed stride for the whole iteration. */
    dst_stride = NpyIter_GetInnerStrideArray(dst_iter)[0];
    dst_countptr = NpyIter_GetInnerLoopSizePtr(dst_iter);

    src_iternext = NpyIter_GetIterNext(src_iter, NULL);
    src_dataptr = NpyIter_GetDataPtrArray(src_iter);
    /* The inner stride is also the fixed stride for the whole iteration. */
    src_stride = NpyIter_GetInnerStrideArray(src_iter)[0];
    src_countptr = NpyIter_GetInnerLoopSizePtr(src_iter);

    if (dst_iternext == NULL || src_iternext == NULL) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }

    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(
                    IsUintAligned(src) && IsAligned(src) &&
                    IsUintAligned(dst) && IsAligned(dst),
                    src_stride, dst_stride,
                    PyArray_DESCR(src), PyArray_DESCR(dst),
                    0,
                    &cast_info, &flags) != NPY_SUCCEED) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }
    /* No need to worry about API use in unbuffered iterator */
    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)src_iter);
    }
    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    dst_count = *dst_countptr;
    src_count = *src_countptr;
    char *args[2] = {src_dataptr[0], dst_dataptr[0]};
    npy_intp strides[2] = {src_stride, dst_stride};

    int res = 0;
    for(;;) {
        /* Transfer the biggest amount that fits both */
        count = (src_count < dst_count) ? src_count : dst_count;
        if (cast_info.func(&cast_info.context,
                args, &count, strides, cast_info.auxdata) < 0) {
            break;
        }

        /* If we exhausted the dst block, refresh it */
        if (dst_count == count) {
            res = dst_iternext(dst_iter);
            if (res == 0) {
                break;
            }
            dst_count = *dst_countptr;
            args[1] = dst_dataptr[0];
        }
        else {
            dst_count -= count;
            args[1] += count*dst_stride;
        }

        /* If we exhausted the src block, refresh it */
        if (src_count == count) {
            res = src_iternext(src_iter);
            if (res == 0) {
                break;
            }
            src_count = *src_countptr;
            args[0] = src_dataptr[0];
        }
        else {
            src_count -= count;
            args[0] += count*src_stride;
        }
    }

    NPY_END_THREADS;

    NPY_cast_info_xfree(&cast_info);
    if (!NpyIter_Deallocate(dst_iter)) {
        res = -1;
    }
    if (!NpyIter_Deallocate(src_iter)) {
        res = -1;
    }

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier((char *)&src_iter);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return res;
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
    return PyArray_CopyAsFlat(dst, src, NPY_CORDER);
}

/*NUMPY_API
 * Copy an Array into another array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dst, PyArrayObject *src)
{
    return PyArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}

/*NUMPY_API
 * PyArray_CheckAxis
 *
 * check that axis is valid
 * convert 0-d arrays to 1-d arrays
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    PyObject *temp1, *temp2;
    int n = PyArray_NDIM(arr);

    if (*axis == NPY_RAVEL_AXIS || n == 0) {
        if (n != 1) {
            temp1 = PyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            if (*axis == NPY_RAVEL_AXIS) {
                *axis = PyArray_NDIM((PyArrayObject *)temp1)-1;
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
    n = PyArray_NDIM((PyArrayObject *)temp2);
    if (check_and_adjust_axis(axis, n) < 0) {
        Py_DECREF(temp2);
        return NULL;
    }
    return temp2;
}


/*NUMPY_API
 * Zeros
 *
 * steals a reference to type. On failure or when PyDataType_SUBARRAY(dtype) is
 * true, dtype will be decrefed.
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    npy_dtype_info dt_info = {NULL, NULL};

    int res = PyArray_ExtractDTypeAndDescriptor(
        type, &dt_info.descr, &dt_info.dtype);

    // steal reference
    Py_XDECREF(type);

    if (res < 0) {
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    PyObject *ret = PyArray_Zeros_int(nd, dims, dt_info.descr, dt_info.dtype,
                                      is_f_order);

    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);

    return ret;
}

/*
 *  Internal version of PyArray_Zeros that accepts a dtypemeta.
 *  Borrows references to the descriptor and dtype.
 */

NPY_NO_EXPORT PyObject *
PyArray_Zeros_int(int nd, npy_intp const *dims, PyArray_Descr *descr,
                  PyArray_DTypeMeta *dtype, int is_f_order)
{
    PyObject *ret = NULL;

    if (descr == NULL) {
        descr = _infer_descr_from_dtype(dtype);
        if (descr == NULL) {
            return NULL;
        }
    }

    /*
     * PyArray_NewFromDescr_int steals a ref to descr,
     * incref so caller of this function can clean up descr
     */
    Py_INCREF(descr);
    ret = PyArray_NewFromDescr_int(
            &PyArray_Type, descr,
            nd, dims, NULL, NULL,
            is_f_order, NULL, NULL,
            _NPY_ARRAY_ZEROED);

    return ret;
}


/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals a reference to type
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    npy_dtype_info dt_info = {NULL, NULL};

    int res = PyArray_ExtractDTypeAndDescriptor(
        type, &dt_info.descr, &dt_info.dtype);

    // steal reference
    Py_XDECREF(type);

    if (res < 0) {
        return NULL;
    }

    PyObject *ret = PyArray_Empty_int(
        nd, dims, dt_info.descr, dt_info.dtype, is_f_order);

    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return ret;
}

/*
 *  Internal version of PyArray_Empty that accepts a dtypemeta.
 *  Borrows references to the descriptor and dtype.
 */

NPY_NO_EXPORT PyObject *
PyArray_Empty_int(int nd, npy_intp const *dims, PyArray_Descr *descr,
                  PyArray_DTypeMeta *dtype, int is_f_order)
{
    PyArrayObject *ret;

    if (descr == NULL) {
        descr = _infer_descr_from_dtype(dtype);
        if (descr == NULL) {
            return NULL;
        }
    }

    /*
     * PyArray_NewFromDescr steals a ref to descr,
     * incref so caller of this function can clean up descr
     */
    Py_INCREF(descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                descr, nd, dims,
                                                NULL, NULL,
                                                is_f_order, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* Logic shared by `empty`, `empty_like`, and `ndarray.__new__` */
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        if (PyArray_SetObjectsToNone(ret) < 0) {
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
static npy_intp
_arange_safe_ceil_to_intp(double value)
{
    double ivalue;

    ivalue = npy_ceil(value);
    /* condition inverted to handle NaN */
    if (npy_isnan(ivalue)) {
        PyErr_SetString(PyExc_ValueError,
            "arange: cannot compute length");
        return -1;
    }
    if (!((double)NPY_MIN_INTP <= ivalue && ivalue <= (double)NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
        return -1;
    }

    return (npy_intp)ivalue;
}


/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    npy_intp length;
    PyArrayObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *obj;
    int ret;
    double delta, tmp_len;
    NPY_BEGIN_THREADS_DEF;

    delta = stop - start;
    tmp_len = delta/step;

    /* Underflow and divide-by-inf check */
    if (tmp_len == 0.0 && delta != 0.0) {
        if (npy_signbit(tmp_len)) {
            length = 0;
        }
        else {
            length = 1;
        }
    }
    else {
        length = _arange_safe_ceil_to_intp(tmp_len);
        if (error_converting(length)) {
            return NULL;
        }
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);
    }
    range = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &length, type_num,
                        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyDataType_GetArrFuncs(PyArray_DESCR(range));

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_DATA(range), range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return (PyObject *)range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                         range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return (PyObject *)range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError,
                "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    funcs->fill(PyArray_DATA(range), length, range);
    NPY_END_THREADS;
    if (PyErr_Occurred()) {
        goto fail;
    }
    return (PyObject *)range;

 fail:
    Py_DECREF(range);
    return NULL;
}

/*
 * the formula is len = (intp) ceil((stop - start) / step);
 */
static npy_intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    npy_intp len, tmp;
    PyObject *zero, *val;
    int next_is_nonzero, val_is_zero;
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

    zero = PyLong_FromLong(0);
    if (!zero) {
        Py_DECREF(*next);
        *next = NULL;
        return -1;
    }

    next_is_nonzero = PyObject_RichCompareBool(*next, zero, Py_NE);
    if (next_is_nonzero == -1) {
        Py_DECREF(zero);
        Py_DECREF(*next);
        *next = NULL;
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;

    if (!val) {
        Py_DECREF(zero);
        return -1;
    }

    val_is_zero = PyObject_RichCompareBool(val, zero, Py_EQ);
    Py_DECREF(zero);
    if (val_is_zero == -1) {
        Py_DECREF(val);
        return -1;
    }

    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        len = _arange_safe_ceil_to_intp(value);
        if (error_converting(len)) {
            Py_DECREF(val);
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        tmp = _arange_safe_ceil_to_intp(value);
        if (error_converting(tmp)) {
            return -1;
        }
        len = PyArray_MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }

        /* Underflow and divide-by-inf check */
        if (val_is_zero && next_is_nonzero) {
            if (npy_signbit(value)) {
                len = 0;
            }
            else {
                len = 1;
            }
        }
        else {
            len = _arange_safe_ceil_to_intp(value);
            if (error_converting(len)) {
                return -1;
            }
        }
    }

    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!*next) {
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
    PyArrayObject *range = NULL;
    PyArray_ArrFuncs *funcs;
    PyObject *next = NULL;
    PyArray_Descr *native = NULL;
    npy_intp length;
    int swap;
    NPY_BEGIN_THREADS_DEF;

    /* Datetime arange is handled specially */
    if ((dtype != NULL && (dtype->type_num == NPY_DATETIME ||
                           dtype->type_num == NPY_TIMEDELTA)) ||
            (dtype == NULL && (is_any_numpy_datetime_or_timedelta(start) ||
                              is_any_numpy_datetime_or_timedelta(stop) ||
                              is_any_numpy_datetime_or_timedelta(step)))) {
        return (PyObject *)datetime_arange(start, stop, step, dtype);
    }

    /* We need to replace many of these, so hold on for easier cleanup */
    Py_XINCREF(start);
    Py_XINCREF(stop);
    Py_XINCREF(step);
    Py_XINCREF(dtype);

    if (!dtype) {
        /* intentionally made to be at least NPY_LONG */
        dtype = PyArray_DescrFromType(NPY_INTP);
        Py_SETREF(dtype, PyArray_DescrFromObject(start, dtype));
        if (dtype == NULL) {
            goto fail;
        }
        if (stop && stop != Py_None) {
            Py_SETREF(dtype, PyArray_DescrFromObject(stop, dtype));
            if (dtype == NULL) {
                goto fail;
            }
        }
        if (step && step != Py_None) {
            Py_SETREF(dtype, PyArray_DescrFromObject(step, dtype));
            if (dtype == NULL) {
                goto fail;
            }
        }
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    if (!PyArray_ISNBO(dtype->byteorder)) {
        native = PyArray_DescrNewByteorder(dtype, NPY_NATBYTE);
        if (native == NULL) {
            goto fail;
        }
        swap = 1;
    }
    else {
        Py_INCREF(dtype);
        native = dtype;
        swap = 0;
    }

    funcs = PyDataType_GetArrFuncs(native);
    if (!funcs->fill) {
        /* This effectively forbids subarray types as well... */
        PyErr_Format(PyExc_TypeError,
                "arange() not supported for inputs with DType %S.",
                Py_TYPE(dtype));
        goto fail;
    }

    if (!step || step == Py_None) {
        Py_XSETREF(step, PyLong_FromLong(1));
        if (step == NULL) {
            goto fail;
        }
    }
    if (!stop || stop == Py_None) {
        Py_XSETREF(stop, start);
        start = PyLong_FromLong(0);
        if (start == NULL) {
            goto fail;
        }
    }

    /* calculate the length and next = start + step*/
    length = _calc_length(start, stop, step, &next,
                          PyTypeNum_ISCOMPLEX(dtype->type_num));
    PyObject *err = PyErr_Occurred();
    if (err) {
        if (PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    if (length <= 0) {
        length = 0;
    }

    Py_INCREF(native);
    range = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &length, native);
    if (range == NULL) {
        goto fail;
    }

    if (length == 0) {
        goto finish;
    }
    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    if (funcs->setitem(start, PyArray_DATA(range), range) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       range) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    funcs->fill(PyArray_DATA(range), length, range);
    NPY_END_THREADS;
    if (PyErr_Occurred()) {
        goto fail;
    }
 finish:
    /* TODO: This swapping could be handled on the fly by the nditer */
    if (swap) {
        PyObject *new;
        new = PyArray_Byteswap(range, 1);
        if (new == NULL) {
            goto fail;
        }
        Py_DECREF(new);
        /* Replace dtype after swapping in-place above: */
        Py_DECREF(PyArray_DESCR(range));
        Py_INCREF(dtype);
        ((PyArrayObject_fields *)range)->descr = dtype;
    }
    Py_DECREF(dtype);
    Py_DECREF(native);
    Py_DECREF(start);
    Py_DECREF(stop);
    Py_DECREF(step);
    Py_XDECREF(next);
    return (PyObject *)range;

 fail:
    Py_XDECREF(dtype);
    Py_XDECREF(native);
    Py_XDECREF(start);
    Py_XDECREF(stop);
    Py_XDECREF(step);
    Py_XDECREF(next);
    Py_XDECREF(range);
    return NULL;
}

/* This array creation function does not steal the reference to dtype. */
static PyArrayObject *
array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, npy_intp num, size_t *nread)
{
    PyArrayObject *r;
    npy_off_t start, numbytes;
    int elsize;

    if (num < 0) {
        int fail = 0;
        start = npy_ftell(fp);
        if (start < 0) {
            fail = 1;
        }
        if (npy_fseek(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = npy_ftell(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (npy_fseek(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
        if (fail) {
            PyErr_SetString(PyExc_OSError,
                            "could not seek in file");
            return NULL;
        }
        num = numbytes / dtype->elsize;
    }

    /*
     * Array creation may move sub-array dimensions from the dtype to array
     * dimensions, so we need to use the original element size when reading.
     */
    elsize = dtype->elsize;

    Py_INCREF(dtype);  /* do not steal the original dtype. */
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &num,
                                              NULL, NULL, 0, NULL);
    if (r == NULL) {
        return NULL;
    }

    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(PyArray_DATA(r), elsize, num, fp);
    NPY_END_ALLOW_THREADS;
    return r;
}

/*
 * Create an array by reading from the given stream, using the passed
 * next_element and skip_separator functions.
 * Does not steal the reference to dtype.
 */
#define FROM_BUFFER_SIZE 4096
static PyArrayObject *
array_from_text(PyArray_Descr *dtype, npy_intp num, char const *sep, size_t *nread,
                void *stream, next_element next, skip_separator skip_sep,
                void *stream_data)
{
    PyArrayObject *r;
    npy_intp i;
    char *dptr, *clean_sep, *tmp;
    int err = 0;
    int stop_reading_flag = 0;  /* -1 means end reached; -2 a parsing error */
    npy_intp thisbuf = 0;
    npy_intp size;
    npy_intp bytes, totalbytes;

    size = (num >= 0) ? num : FROM_BUFFER_SIZE;

    /*
     * Array creation may move sub-array dimensions from the dtype to array
     * dimensions, so we need to use the original dtype when reading.
     */
    Py_INCREF(dtype);

    r = (PyArrayObject *)
        PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &size,
                             NULL, NULL, 0, NULL);
    if (r == NULL) {
        return NULL;
    }

    clean_sep = swab_separator(sep);
    if (clean_sep == NULL) {
        err = 1;
        goto fail;
    }

    NPY_BEGIN_ALLOW_THREADS;
    totalbytes = bytes = size * dtype->elsize;
    dptr = PyArray_DATA(r);
    for (i = 0; num < 0 || i < num; i++) {
        stop_reading_flag = next(&stream, dptr, dtype, stream_data);
        if (stop_reading_flag < 0) {
            break;
        }
        *nread += 1;
        thisbuf += 1;
        dptr += dtype->elsize;
        if (num < 0 && thisbuf == size) {
            totalbytes += bytes;
            /* The handler is always valid */
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), totalbytes,
                                  PyArray_HANDLER(r));
            if (tmp == NULL) {
                err = 1;
                break;
            }
            ((PyArrayObject_fields *)r)->data = tmp;
            dptr = tmp + (totalbytes - bytes);
            thisbuf = 0;
        }
        stop_reading_flag = skip_sep(&stream, clean_sep, stream_data);
        if (stop_reading_flag < 0) {
            if (num == i + 1) {
                /* if we read as much as requested sep is optional */
                stop_reading_flag = -1;
            }
            break;
        }
    }
    if (num < 0) {
        const size_t nsize = PyArray_MAX(*nread,1)*dtype->elsize;

        if (nsize != 0) {
            /* The handler is always valid */
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), nsize,
                                  PyArray_HANDLER(r));
            if (tmp == NULL) {
                err = 1;
            }
            else {
                PyArray_DIMS(r)[0] = *nread;
                ((PyArrayObject_fields *)r)->data = tmp;
            }
        }
    }
    NPY_END_ALLOW_THREADS;

    free(clean_sep);

    if (stop_reading_flag == -2) {
        if (PyErr_Occurred()) {
            /* If an error is already set (unlikely), do not create new one */
            Py_DECREF(r);
            return NULL;
        }
        PyErr_SetString(PyExc_ValueError,
            "string or file could not be read to its end due to unmatched data");
        goto fail;
    }

fail:
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
 * The reference to `dtype` is stolen (it is possible that the passed in
 * dtype is not held on to).
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

    if (dtype == NULL) {
        return NULL;
    }

    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        /* Nothing to read, just create an empty array of the requested type */
        return PyArray_NewFromDescr_int(
                &PyArray_Type, dtype,
                1, &num, NULL, NULL,
                0, NULL, NULL,
                _NPY_ARRAY_ALLOW_EMPTY_STRING);
    }
    if ((sep == NULL) || (strlen(sep) == 0)) {
        ret = array_fromfile_binary(fp, dtype, num, &nread);
    }
    else {
        if (PyDataType_GetArrFuncs(dtype)->scanfunc == NULL) {
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
        /*
         * Realloc memory for smaller number of elements, use original dtype
         * which may have include a subarray (and is used for `nread`).
         */
        const size_t nsize = PyArray_MAX(nread,1) * dtype->elsize;
        char *tmp;

        /* The handler is always valid */
        if((tmp = PyDataMem_UserRENEW(PyArray_DATA(ret), nsize,
                                     PyArray_HANDLER(ret))) == NULL) {
            Py_DECREF(dtype);
            Py_DECREF(ret);
            return PyErr_NoMemory();
        }
        ((PyArrayObject_fields *)ret)->data = tmp;
        PyArray_DIMS(ret)[0] = nread;
    }
    Py_DECREF(dtype);
    return (PyObject *)ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   npy_intp count, npy_intp offset)
{
    PyArrayObject *ret;
    char *data;
    Py_buffer view;
    Py_ssize_t ts;
    npy_intp s, n;
    int itemsize;
    int writeable = 1;

    if (type == NULL) {
        return NULL;
    }

    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory"\
                        " buffer");
        Py_DECREF(type);
        return NULL;
    }
    if (PyDataType_ISUNSIZED(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;
    }

    /*
     * If the object supports `releasebuffer`, the new buffer protocol allows
     * tying the memories lifetime to the `Py_buffer view`.
     * NumPy cannot hold on to the view itself (it is not an object) so it
     * has to wrap the original object in a Python `memoryview` which deals
     * with the lifetime management for us.
     * For backwards compatibility of `arr.base` we try to avoid this when
     * possible.  (For example, NumPy arrays will never get wrapped here!)
     */
    if (Py_TYPE(buf)->tp_as_buffer
            && Py_TYPE(buf)->tp_as_buffer->bf_releasebuffer) {
        buf = PyMemoryView_FromObject(buf);
        if (buf == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(buf);
    }

    if (PyObject_GetBuffer(buf, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        writeable = 0;
        PyErr_Clear();
        if (PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE) < 0) {
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }
    data = (char *)view.buf;
    ts = view.len;
    /* `buf` is an array or a memoryview; so we know `view` does not own data */
    PyBuffer_Release(&view);

    if ((offset < 0) || (offset > ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be non-negative and no greater than buffer "\
                     "length (%" NPY_INTP_FMT ")", (npy_intp)ts);
        Py_DECREF(buf);
        Py_DECREF(type);
        return NULL;
    }

    data += offset;
    s = (npy_intp)ts - offset;
    n = (npy_intp)count;
    itemsize = type->elsize;
    if (n < 0) {
        if (itemsize == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot determine count if itemsize is 0");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
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

    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, type,
            1, &n, NULL, data,
            NPY_ARRAY_DEFAULT, NULL, buf);
    Py_DECREF(buf);
    if (ret == NULL) {
        return NULL;
    }

    if (!writeable) {
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
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
    npy_bool binary;

    if (dtype == NULL) {
        dtype=PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        if (dtype == NULL) {
            return NULL;
        }
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
        /*
         * NewFromDescr may replace dtype to absorb subarray shape
         * into the array, so get size beforehand.
         */
        npy_intp size_to_copy = num*dtype->elsize;
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, dtype,
                                 1, &num, NULL, NULL,
                                 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
        memcpy(PyArray_DATA(ret), data, size_to_copy);
    }
    else {
        /* read from character-based string */
        size_t nread = 0;
        char *end;

        if (PyDataType_GetArrFuncs(dtype)->fromstr == NULL) {
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
        Py_DECREF(dtype);
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
    PyObject *iter = NULL;
    PyArrayObject *ret = NULL;
    npy_intp i, elsize, elcount;

    if (dtype == NULL) {
        return NULL;
    }

    iter = PyObject_GetIter(obj);
    if (iter == NULL) {
        goto done;
    }

    if (PyDataType_ISUNSIZED(dtype)) {
        /* If this error is removed, the `ret` allocation may need fixing */
        PyErr_SetString(PyExc_ValueError,
                "Must specify length when using variable-size data-type.");
        goto done;
    }
    if (count < 0) {
        elcount = PyObject_LengthHint(obj, 0);
        if (elcount < 0) {
            goto done;
        }
    }
    else {
        elcount = count;
    }

    elsize = dtype->elsize;

    /*
     * Note that PyArray_DESCR(ret) may not match dtype.  There are exactly
     * two cases where this can happen: empty strings/bytes/void (rejected
     * above) and subarray dtypes (supported by sticking with `dtype`).
     */
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1,
                                                &elcount, NULL,NULL, 0, NULL);
    if (ret == NULL) {
        goto done;
    }

    char *item = PyArray_BYTES(ret);
    for (i = 0; i < count || count == -1; i++, item += elsize) {
        PyObject *value = PyIter_Next(iter);
        if (value == NULL) {
            if (PyErr_Occurred()) {
                /* Fetching next item failed perhaps due to exhausting iterator */
                goto done;
            }
            break;
        }

        if (NPY_UNLIKELY(i >= elcount) && elsize != 0) {
            char *new_data = NULL;
            npy_intp nbytes;
            /*
              Grow PyArray_DATA(ret):
              this is similar for the strategy for PyListObject, but we use
              50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
              TODO: The loadtxt code now uses a `growth` helper that would
                    be suitable to reuse here.
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            if (!npy_mul_sizes_with_overflow(&nbytes, elcount, elsize)) {
                /* The handler is always valid */
                new_data = PyDataMem_UserRENEW(
                        PyArray_BYTES(ret), nbytes, PyArray_HANDLER(ret));
            }
            if (new_data == NULL) {
                PyErr_SetString(PyExc_MemoryError,
                        "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            ((PyArrayObject_fields *)ret)->data = new_data;
            /* resize array for cleanup: */
            PyArray_DIMS(ret)[0] = elcount;
            /* Reset `item` pointer to point into realloc'd chunk */
            item = new_data + i * elsize;
            if (PyDataType_FLAGCHK(dtype, NPY_NEEDS_INIT)) {
                /* Initialize new chunk: */
                memset(item, 0, nbytes - i * elsize);
            }
        }

        if (PyArray_Pack(dtype, item, value) < 0) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }

    if (i < count) {
        PyErr_Format(PyExc_ValueError,
                "iterator too short: Expected %zd but iterator had only %zd "
                "items.", (Py_ssize_t)count, (Py_ssize_t)i);
        goto done;
    }

    /*
     * Realloc the data so that don't keep extra memory tied up and fix
     * the arrays first dimension (there could be more than one).
     */
    if (i == 0 || elsize == 0) {
        /* The size cannot be zero for realloc. */
    }
    else {
        /* Resize array to actual final size (it may be too large) */
        /* The handler is always valid */
        char *new_data = PyDataMem_UserRENEW(
                PyArray_DATA(ret), i * elsize, PyArray_HANDLER(ret));

        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate array memory");
            goto done;
        }
        ((PyArrayObject_fields *)ret)->data = new_data;

        if (count < 0) {
            /*
             * If the count was smaller than zero, the strides may be all 0
             * (even in the later dimensions for `count < 0`!
             * Thus, fix all strides here again for C-contiguity.
             */
            int oflags;
            _array_fill_strides(
                    PyArray_STRIDES(ret), PyArray_DIMS(ret), PyArray_NDIM(ret),
                    PyArray_ITEMSIZE(ret), NPY_ARRAY_C_CONTIGUOUS, &oflags);
            PyArray_STRIDES(ret)[0] = elsize;
            assert(oflags & NPY_ARRAY_C_CONTIGUOUS);
        }
    }
    PyArray_DIMS(ret)[0] = i;

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
 * and the NPY_ARRAY_C_CONTIGUOUS bit will be set.  If the flags argument
 * has the NPY_ARRAY_F_CONTIGUOUS bit set, then a FORTRAN-style strides array will be
 * created (and of course the NPY_ARRAY_F_CONTIGUOUS flag bit will be set).
 *
 * If data is not given but created here, then flags will be NPY_ARRAY_DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 *
 * Dimensions and itemsize must have been checked for validity.
 */

NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
    npy_bool not_cf_contig = 0;
    npy_bool nod = 0; /* A dim != 1 was found */

    /* Check if new array is both F- and C-contiguous */
    for (i = 0; i < nd; i++) {
        if (dims[i] != 1) {
            if (nod) {
                not_cf_contig = 1;
                break;
            }
            nod = 1;
        }
    }

    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS)) ==
                                            NPY_ARRAY_F_CONTIGUOUS) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
            else {
                not_cf_contig = 0;
            }
        }
        if (not_cf_contig) {
            *objflags = ((*objflags)|NPY_ARRAY_F_CONTIGUOUS) &
                                            ~NPY_ARRAY_C_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS);
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
            else {
                not_cf_contig = 0;
            }
        }
        if (not_cf_contig) {
            *objflags = ((*objflags)|NPY_ARRAY_C_CONTIGUOUS) &
                                            ~NPY_ARRAY_F_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    return;
}

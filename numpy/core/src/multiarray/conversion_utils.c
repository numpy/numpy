#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"

#include "common.h"
#include "arraytypes.h"

#include "conversion_utils.h"
#include "alloc.h"
#include "npy_buffer.h"

static int
PyArray_PyIntAsInt_ErrMsg(PyObject *o, const char * msg) NPY_GCC_NONNULL(2);
static npy_intp
PyArray_PyIntAsIntp_ErrMsg(PyObject *o, const char * msg) NPY_GCC_NONNULL(2);

/****************************************************************
* Useful function for conversion when used with PyArg_ParseTuple
****************************************************************/

/*NUMPY_API
 *
 * Useful to pass as converter function for O& processing in PyArgs_ParseTuple.
 *
 * This conversion function can be used with the "O&" argument for
 * PyArg_ParseTuple.  It will immediately return an object of array type
 * or will convert to a NPY_ARRAY_CARRAY any other object.
 *
 * If you use PyArray_Converter, you must DECREF the array when finished
 * as you get a new reference to it.
 */
NPY_NO_EXPORT int
PyArray_Converter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    }
    else {
        *address = PyArray_FROM_OF(object, NPY_ARRAY_CARRAY);
        if (*address == NULL) {
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}

/*NUMPY_API
 * Useful to pass as converter function for O& processing in
 * PyArgs_ParseTuple for output arrays
 */
NPY_NO_EXPORT int
PyArray_OutputConverter(PyObject *object, PyArrayObject **address)
{
    if (object == NULL || object == Py_None) {
        *address = NULL;
        return NPY_SUCCEED;
    }
    if (PyArray_Check(object)) {
        *address = (PyArrayObject *)object;
        return NPY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "output must be an array");
        *address = NULL;
        return NPY_FAIL;
    }
}

/*NUMPY_API
 * Get intp chunk from sequence
 *
 * This function takes a Python sequence object and allocates and
 * fills in an intp array with the converted values.
 *
 * Remember to free the pointer seq.ptr when done using
 * PyDimMem_FREE(seq.ptr)**
 */
NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    Py_ssize_t len;
    int nd;

    seq->ptr = NULL;
    seq->len = 0;
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }
    len = PySequence_Size(obj);
    if (len == -1) {
        /* Check to see if it is an integer number */
        if (PyNumber_Check(obj)) {
            /*
             * After the deprecation the PyNumber_Check could be replaced
             * by PyIndex_Check.
             * FIXME 1.9 ?
             */
            len = 1;
        }
    }
    if (len < 0) {
        PyErr_SetString(PyExc_TypeError,
                "expected sequence object with len >= 0 or a single integer");
        return NPY_FAIL;
    }
    if (len > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError, "maximum supported dimension for an ndarray is %d"
                     ", found %d", NPY_MAXDIMS, len);
        return NPY_FAIL;
    }
    if (len > 0) {
        seq->ptr = npy_alloc_cache_dim(len);
        if (seq->ptr == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
    }
    seq->len = len;
    nd = PyArray_IntpFromIndexSequence(obj, (npy_intp *)seq->ptr, len);
    if (nd == -1 || nd != len) {
        npy_free_cache_dim_obj(*seq);
        seq->ptr = NULL;
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/*
 * Like PyArray_IntpConverter, but leaves `seq` untouched if `None` is passed
 * rather than treating `None` as `()`.
 */
NPY_NO_EXPORT int
PyArray_OptionalIntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }

    return PyArray_IntpConverter(obj, seq);
}

/*NUMPY_API
 * Get buffer chunk from object
 *
 * this function takes a Python object which exposes the (single-segment)
 * buffer interface and returns a pointer to the data segment
 *
 * You should increment the reference count by one of buf->base
 * if you will hang on to a reference
 *
 * You only get a borrowed reference to the object. Do not free the
 * memory...
 */
NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf)
{
    Py_buffer view;

    buf->ptr = NULL;
    buf->flags = NPY_ARRAY_BEHAVED;
    buf->base = NULL;
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }

    if (PyObject_GetBuffer(obj, &view,
                PyBUF_ANY_CONTIGUOUS|PyBUF_WRITABLE|PyBUF_SIMPLE) != 0) {
        PyErr_Clear();
        buf->flags &= ~NPY_ARRAY_WRITEABLE;
        if (PyObject_GetBuffer(obj, &view,
                PyBUF_ANY_CONTIGUOUS|PyBUF_SIMPLE) != 0) {
            return NPY_FAIL;
        }
    }

    buf->ptr = view.buf;
    buf->len = (npy_intp) view.len;

    /*
     * In Python 3 both of the deprecated functions PyObject_AsWriteBuffer and
     * PyObject_AsReadBuffer that this code replaces release the buffer. It is
     * up to the object that supplies the buffer to guarantee that the buffer
     * sticks around after the release.
     */
    PyBuffer_Release(&view);
    _dealloc_cached_buffer_info(obj);

    /* Point to the base of the buffer object if present */
    if (PyMemoryView_Check(obj)) {
        buf->base = PyMemoryView_GET_BASE(obj);
    }
    if (buf->base == NULL) {
        buf->base = obj;
    }
    return NPY_SUCCEED;
}

/*NUMPY_API
 * Get axis from an object (possibly None) -- a converter function,
 *
 * See also PyArray_ConvertMultiAxis, which also handles a tuple of axes.
 */
NPY_NO_EXPORT int
PyArray_AxisConverter(PyObject *obj, int *axis)
{
    if (obj == Py_None) {
        *axis = NPY_MAXDIMS;
    }
    else {
        *axis = PyArray_PyIntAsInt_ErrMsg(obj,
                               "an integer is required for the axis");
        if (error_converting(*axis)) {
            return NPY_FAIL;
        }
    }
    return NPY_SUCCEED;
}

/*
 * Converts an axis parameter into an ndim-length C-array of
 * boolean flags, True for each axis specified.
 *
 * If obj is None or NULL, everything is set to True. If obj is a tuple,
 * each axis within the tuple is set to True. If obj is an integer,
 * just that axis is set to True.
 */
NPY_NO_EXPORT int
PyArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags)
{
    /* None means all of the axes */
    if (axis_in == Py_None || axis_in == NULL) {
        memset(out_axis_flags, 1, ndim);
        return NPY_SUCCEED;
    }
    /* A tuple of which axes */
    else if (PyTuple_Check(axis_in)) {
        int i, naxes;

        memset(out_axis_flags, 0, ndim);

        naxes = PyTuple_Size(axis_in);
        if (naxes < 0) {
            return NPY_FAIL;
        }
        for (i = 0; i < naxes; ++i) {
            PyObject *tmp = PyTuple_GET_ITEM(axis_in, i);
            int axis = PyArray_PyIntAsInt_ErrMsg(tmp,
                          "integers are required for the axis tuple elements");
            if (error_converting(axis)) {
                return NPY_FAIL;
            }
            if (check_and_adjust_axis(&axis, ndim) < 0) {
                return NPY_FAIL;
            }
            if (out_axis_flags[axis]) {
                PyErr_SetString(PyExc_ValueError,
                        "duplicate value in 'axis'");
                return NPY_FAIL;
            }
            out_axis_flags[axis] = 1;
        }

        return NPY_SUCCEED;
    }
    /* Try to interpret axis as an integer */
    else {
        int axis;

        memset(out_axis_flags, 0, ndim);

        axis = PyArray_PyIntAsInt_ErrMsg(axis_in,
                                   "an integer is required for the axis");

        if (error_converting(axis)) {
            return NPY_FAIL;
        }
        /*
         * Special case letting axis={-1,0} slip through for scalars,
         * for backwards compatibility reasons.
         */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            return NPY_SUCCEED;
        }

        if (check_and_adjust_axis(&axis, ndim) < 0) {
            return NPY_FAIL;
        }

        out_axis_flags[axis] = 1;

        return NPY_SUCCEED;
    }
}

/*NUMPY_API
 * Convert an object to true / false
 */
NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, npy_bool *val)
{
    if (PyObject_IsTrue(object)) {
        *val = NPY_TRUE;
    }
    else {
        *val = NPY_FALSE;
    }
    if (PyErr_Occurred()) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

static int
string_converter_helper(
    PyObject *object,
    void *out,
    int (*str_func)(char const*, Py_ssize_t, void*),
    char const *name,
    char const *message)
{
    /* allow bytes for compatibility */
    PyObject *str_object = NULL;
    if (PyBytes_Check(object)) {
        str_object = PyUnicode_FromEncodedObject(object, NULL, NULL);
        if (str_object == NULL) {
            PyErr_Format(PyExc_ValueError,
                "%s %s (got %R)", name, message, object);
            return NPY_FAIL;
        }
    }
    else if (PyUnicode_Check(object)) {
        str_object = object;
        Py_INCREF(str_object);
    }
    else {
        PyErr_Format(PyExc_TypeError,
            "%s must be str, not %s", name, Py_TYPE(object)->tp_name);
        return NPY_FAIL;
    }

    Py_ssize_t length;
    char const *str = PyUnicode_AsUTF8AndSize(str_object, &length);
    if (str == NULL) {
        Py_DECREF(str_object);
        return NPY_FAIL;
    }

    int ret = str_func(str, length, out);
    Py_DECREF(str_object);
    if (ret < 0) {
            PyErr_Format(PyExc_ValueError,
                "%s %s (got %R)", name, message, object);
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

static int byteorder_parser(char const *str, Py_ssize_t length, void *data)
{
    char *endian = (char *)data;

    if (length < 1) {
        return -1;
    }
    else if (str[0] == NPY_BIG || str[0] == NPY_LITTLE
        || str[0] == NPY_NATIVE || str[0] == NPY_IGNORE) {
        *endian = str[0];
        return 0;
    }
    else if (str[0] == 'b' || str[0] == 'B') {
        *endian = NPY_BIG;
        return 0;
    }
    else if (str[0] == 'l' || str[0] == 'L') {
        *endian = NPY_LITTLE;
        return 0;
    }
    else if (str[0] == 'n' || str[0] == 'N') {
        *endian = NPY_NATIVE;
        return 0;
    }
    else if (str[0] == 'i' || str[0] == 'I') {
        *endian = NPY_IGNORE;
        return 0;
    }
    else if (str[0] == 's' || str[0] == 'S') {
        *endian = NPY_SWAP;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert object to endian
 */
NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian)
{
    return string_converter_helper(
        obj, (void *)endian, byteorder_parser, "byteorder", "not recognized");
}

static int sortkind_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_SORTKIND *sortkind = (NPY_SORTKIND *)data;

    if (length < 1) {
        return -1;
    }
    if (str[0] == 'q' || str[0] == 'Q') {
        *sortkind = NPY_QUICKSORT;
        return 0;
    }
    else if (str[0] == 'h' || str[0] == 'H') {
        *sortkind = NPY_HEAPSORT;
        return 0;
    }
    else if (str[0] == 'm' || str[0] == 'M') {
        /*
         * Mergesort is an alias for NPY_STABLESORT.
         * That maintains backwards compatibility while
         * allowing other types of stable sorts to be used.
         */
        *sortkind = NPY_MERGESORT;
        return 0;
    }
    else if (str[0] == 's' || str[0] == 'S') {
        /*
         * NPY_STABLESORT is one of
         *
         *   - mergesort
         *   - timsort
         *
         *  Which one is used depends on the data type.
         */
        *sortkind = NPY_STABLESORT;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert object to sort kind
 */
NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind)
{
    /* Leave the desired default from the caller for Py_None */
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }
    return string_converter_helper(
        obj, (void *)sortkind, sortkind_parser, "sort kind",
        "must be one of 'quick', 'heap', or 'stable'");
}

static int selectkind_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_SELECTKIND *selectkind = (NPY_SELECTKIND *)data;

    if (length == 11 && strcmp(str, "introselect") == 0) {
        *selectkind = NPY_INTROSELECT;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert object to select kind
 */
NPY_NO_EXPORT int
PyArray_SelectkindConverter(PyObject *obj, NPY_SELECTKIND *selectkind)
{
    return string_converter_helper(
        obj, (void *)selectkind, selectkind_parser, "select kind",
        "must be 'introselect'");
}

static int searchside_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_SEARCHSIDE *side = (NPY_SEARCHSIDE *)data;

    if (length < 1) {
        return -1;
    }
    else if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
        return 0;
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert object to searchsorted side
 */
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr)
{
    return string_converter_helper(
        obj, addr, searchside_parser, "search side",
        "must be 'left' or 'right'");
}

static int order_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_ORDER *val = (NPY_ORDER *)data;
    if (length != 1) {
        return -1;
    }
    if (str[0] == 'C' || str[0] == 'c') {
        *val = NPY_CORDER;
        return 0;
    }
    else if (str[0] == 'F' || str[0] == 'f') {
        *val = NPY_FORTRANORDER;
        return 0;
    }
    else if (str[0] == 'A' || str[0] == 'a') {
        *val = NPY_ANYORDER;
        return 0;
    }
    else if (str[0] == 'K' || str[0] == 'k') {
        *val = NPY_KEEPORDER;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert an object to FORTRAN / C / ANY / KEEP
 */
NPY_NO_EXPORT int
PyArray_OrderConverter(PyObject *object, NPY_ORDER *val)
{
    /* Leave the desired default from the caller for Py_None */
    if (object == Py_None) {
        return NPY_SUCCEED;
    }
    return string_converter_helper(
        object, (void *)val, order_parser, "order",
        "must be one of 'C', 'F', 'A', or 'K'");
}

static int clipmode_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_CLIPMODE *val = (NPY_CLIPMODE *)data;
    if (length < 1) {
        return -1;
    }
    if (str[0] == 'C' || str[0] == 'c') {
        *val = NPY_CLIP;
        return 0;
    }
    else if (str[0] == 'W' || str[0] == 'w') {
        *val = NPY_WRAP;
        return 0;
    }
    else if (str[0] == 'R' || str[0] == 'r') {
        *val = NPY_RAISE;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert an object to NPY_RAISE / NPY_CLIP / NPY_WRAP
 */
NPY_NO_EXPORT int
PyArray_ClipmodeConverter(PyObject *object, NPY_CLIPMODE *val)
{
    if (object == NULL || object == Py_None) {
        *val = NPY_RAISE;
    }

    else if (PyBytes_Check(object) || PyUnicode_Check(object)) {
        return string_converter_helper(
            object, (void *)val, clipmode_parser, "clipmode",
            "must be one of 'clip', 'raise', or 'wrap'");
    }
    else {
        /* For users passing `np.RAISE`, `np.WRAP`, `np.CLIP` */
        int number = PyArray_PyIntAsInt(object);
        if (error_converting(number)) {
            goto fail;
        }
        if (number <= (int) NPY_RAISE
                && number >= (int) NPY_CLIP) {
            *val = (NPY_CLIPMODE) number;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "integer clipmode must be np.RAISE, np.WRAP, or np.CLIP");
        }
    }
    return NPY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError,
                    "clipmode not understood");
    return NPY_FAIL;
}

/*NUMPY_API
 * Convert an object to an array of n NPY_CLIPMODE values.
 * This is intended to be used in functions where a different mode
 * could be applied to each axis, like in ravel_multi_index.
 */
NPY_NO_EXPORT int
PyArray_ConvertClipmodeSequence(PyObject *object, NPY_CLIPMODE *modes, int n)
{
    int i;
    /* Get the clip mode(s) */
    if (object && (PyTuple_Check(object) || PyList_Check(object))) {
        if (PySequence_Size(object) != n) {
            PyErr_Format(PyExc_ValueError,
                    "list of clipmodes has wrong length (%zd instead of %d)",
                    PySequence_Size(object), n);
            return NPY_FAIL;
        }

        for (i = 0; i < n; ++i) {
            PyObject *item = PySequence_GetItem(object, i);
            if(item == NULL) {
                return NPY_FAIL;
            }

            if(PyArray_ClipmodeConverter(item, &modes[i]) != NPY_SUCCEED) {
                Py_DECREF(item);
                return NPY_FAIL;
            }

            Py_DECREF(item);
        }
    }
    else if (PyArray_ClipmodeConverter(object, &modes[0]) == NPY_SUCCEED) {
        for (i = 1; i < n; ++i) {
            modes[i] = modes[0];
        }
    }
    else {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

static int casting_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_CASTING *casting = (NPY_CASTING *)data;
    if (length < 2) {
        return -1;
    }
    switch (str[2]) {
    case 0:
        if (length == 2 && strcmp(str, "no") == 0) {
            *casting = NPY_NO_CASTING;
            return 0;
        }
        break;
    case 'u':
        if (length == 5 && strcmp(str, "equiv") == 0) {
            *casting = NPY_EQUIV_CASTING;
            return 0;
        }
        break;
    case 'f':
        if (length == 4 && strcmp(str, "safe") == 0) {
            *casting = NPY_SAFE_CASTING;
            return 0;
        }
        break;
    case 'm':
        if (length == 9 && strcmp(str, "same_kind") == 0) {
            *casting = NPY_SAME_KIND_CASTING;
            return 0;
        }
        break;
    case 's':
        if (length == 6 && strcmp(str, "unsafe") == 0) {
            *casting = NPY_UNSAFE_CASTING;
            return 0;
        }
        break;
    }
    return -1;
}

/*NUMPY_API
 * Convert any Python object, *obj*, to an NPY_CASTING enum.
 */
NPY_NO_EXPORT int
PyArray_CastingConverter(PyObject *obj, NPY_CASTING *casting)
{
    return string_converter_helper(
        obj, (void *)casting, casting_parser, "casting",
            "must be one of 'no', 'equiv', 'safe', "
            "'same_kind', or 'unsafe'");
    return 0;
}

/*****************************
* Other conversion functions
*****************************/

static int
PyArray_PyIntAsInt_ErrMsg(PyObject *o, const char * msg)
{
    npy_intp long_value;
    /* This assumes that NPY_SIZEOF_INTP >= NPY_SIZEOF_INT */
    long_value = PyArray_PyIntAsIntp_ErrMsg(o, msg);

#if (NPY_SIZEOF_INTP > NPY_SIZEOF_INT)
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    return (int) long_value;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o)
{
    return PyArray_PyIntAsInt_ErrMsg(o, "an integer is required");
}

static npy_intp
PyArray_PyIntAsIntp_ErrMsg(PyObject *o, const char * msg)
{
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long long long_value = -1;
#else
    long long_value = -1;
#endif
    PyObject *obj, *err;

    /*
     * Be a bit stricter and not allow bools.
     * np.bool_ is also disallowed as Boolean arrays do not currently
     * support index.
     */
    if (!o || PyBool_Check(o) || PyArray_IsScalar(o, Bool)) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }

    /*
     * Since it is the usual case, first check if o is an integer. This is
     * an exact check, since otherwise __index__ is used.
     */
    if (PyLong_CheckExact(o)) {
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
        long_value = PyLong_AsLongLong(o);
#else
        long_value = PyLong_AsLong(o);
#endif
        return (npy_intp)long_value;
    }

    /*
     * The most general case. PyNumber_Index(o) covers everything
     * including arrays. In principle it may be possible to replace
     * the whole function by PyIndex_AsSSize_t after deprecation.
     */
    obj = PyNumber_Index(o);
    if (obj == NULL) {
        return -1;
    }
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long_value = PyLong_AsLongLong(obj);
#else
    long_value = PyLong_AsLong(obj);
#endif
    Py_DECREF(obj);

    if (error_converting(long_value)) {
        err = PyErr_Occurred();
        /* Only replace TypeError's here, which are the normal errors. */
        if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {
            PyErr_SetString(PyExc_TypeError, msg);
        }
        return -1;
    }
    goto overflow_check; /* silence unused warning */

overflow_check:
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
  #if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_INTP)
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#else
  #if (NPY_SIZEOF_LONG > NPY_SIZEOF_INTP)
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#endif
    return long_value;
}

/*NUMPY_API*/
NPY_NO_EXPORT npy_intp
PyArray_PyIntAsIntp(PyObject *o)
{
    return PyArray_PyIntAsIntp_ErrMsg(o, "an integer is required");
}


/*
 * PyArray_IntpFromIndexSequence
 * Returns the number of dimensions or -1 if an error occurred.
 * vals must be large enough to hold maxvals.
 * Opposed to PyArray_IntpFromSequence it uses and returns npy_intp
 * for the number of values.
 */
NPY_NO_EXPORT npy_intp
PyArray_IntpFromIndexSequence(PyObject *seq, npy_intp *vals, npy_intp maxvals)
{
    Py_ssize_t nd;
    npy_intp i;
    PyObject *op, *err;

    /*
     * Check to see if sequence is a single integer first.
     * or, can be made into one
     */
    nd = PySequence_Length(seq);
    if (nd == -1) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }

        vals[0] = PyArray_PyIntAsIntp(seq);
        if(vals[0] == -1) {
            err = PyErr_Occurred();
            if (err &&
                    PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                PyErr_SetString(PyExc_ValueError,
                        "Maximum allowed dimension exceeded");
            }
            if(err != NULL) {
                return -1;
            }
        }
        nd = 1;
    }
    else {
        for (i = 0; i < PyArray_MIN(nd,maxvals); i++) {
            op = PySequence_GetItem(seq, i);
            if (op == NULL) {
                return -1;
            }

            vals[i] = PyArray_PyIntAsIntp(op);
            Py_DECREF(op);
            if(vals[i] == -1) {
                err = PyErr_Occurred();
                if (err &&
                        PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                    PyErr_SetString(PyExc_ValueError,
                            "Maximum allowed dimension exceeded");
                }
                if(err != NULL) {
                    return -1;
                }
            }
        }
    }
    return nd;
}

/*NUMPY_API
 * PyArray_IntpFromSequence
 * Returns the number of integers converted or -1 if an error occurred.
 * vals must be large enough to hold maxvals
 */
NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, npy_intp *vals, int maxvals)
{
    return PyArray_IntpFromIndexSequence(seq, vals, (npy_intp)maxvals);
}


/**
 * WARNING: This flag is a bad idea, but was the only way to both
 *   1) Support unpickling legacy pickles with object types.
 *   2) Deprecate (and later disable) usage of O4 and O8
 *
 * The key problem is that the pickled representation unpickles by
 * directly calling the dtype constructor, which has no way of knowing
 * that it is in an unpickle context instead of a normal context without
 * evil global state like we create here.
 */
NPY_NO_EXPORT int evil_global_disable_warn_O4O8_flag = 0;

/*NUMPY_API
 * Typestr converter
 */
NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype)
{
    int newtype = NPY_NOTYPE;

    switch (gentype) {
        case NPY_GENBOOLLTR:
            if (itemsize == 1) {
                newtype = NPY_BOOL;
            }
            break;

        case NPY_SIGNEDLTR:
            switch(itemsize) {
                case 1:
                    newtype = NPY_INT8;
                    break;
                case 2:
                    newtype = NPY_INT16;
                    break;
                case 4:
                    newtype = NPY_INT32;
                    break;
                case 8:
                    newtype = NPY_INT64;
                    break;
#ifdef NPY_INT128
                case 16:
                    newtype = NPY_INT128;
                    break;
#endif
            }
            break;

        case NPY_UNSIGNEDLTR:
            switch(itemsize) {
                case 1:
                    newtype = NPY_UINT8;
                    break;
                case 2:
                    newtype = NPY_UINT16;
                    break;
                case 4:
                    newtype = NPY_UINT32;
                    break;
                case 8:
                    newtype = NPY_UINT64;
                    break;
#ifdef NPY_INT128
                case 16:
                    newtype = NPY_UINT128;
                    break;
#endif
            }
            break;

        case NPY_FLOATINGLTR:
            switch(itemsize) {
                case 2:
                    newtype = NPY_FLOAT16;
                    break;
                case 4:
                    newtype = NPY_FLOAT32;
                    break;
                case 8:
                    newtype = NPY_FLOAT64;
                    break;
#ifdef NPY_FLOAT80
                case 10:
                    newtype = NPY_FLOAT80;
                    break;
#endif
#ifdef NPY_FLOAT96
                case 12:
                    newtype = NPY_FLOAT96;
                    break;
#endif
#ifdef NPY_FLOAT128
                case 16:
                    newtype = NPY_FLOAT128;
                    break;
#endif
            }
            break;

        case NPY_COMPLEXLTR:
            switch(itemsize) {
                case 8:
                    newtype = NPY_COMPLEX64;
                    break;
                case 16:
                    newtype = NPY_COMPLEX128;
                    break;
#ifdef NPY_FLOAT80
                case 20:
                    newtype = NPY_COMPLEX160;
                    break;
#endif
#ifdef NPY_FLOAT96
                case 24:
                    newtype = NPY_COMPLEX192;
                    break;
#endif
#ifdef NPY_FLOAT128
                case 32:
                    newtype = NPY_COMPLEX256;
                    break;
#endif
            }
            break;

        case NPY_OBJECTLTR:
            /*
             * For 'O4' and 'O8', let it pass, but raise a
             * deprecation warning. For all other cases, raise
             * an exception by leaving newtype unset.
             */
            if (itemsize == 4 || itemsize == 8) {
                int ret = 0;
                if (evil_global_disable_warn_O4O8_flag) {
                    /* 2012-02-04, 1.7, not sure when this can be removed */
                    ret = DEPRECATE("DType strings 'O4' and 'O8' are "
                            "deprecated because they are platform "
                            "specific. Use 'O' instead");
                }

                if (ret == 0) {
                    newtype = NPY_OBJECT;
                }
            }
            break;

        case NPY_STRINGLTR:
        case NPY_STRINGLTR2:
            newtype = NPY_STRING;
            break;

        case NPY_UNICODELTR:
            newtype = NPY_UNICODE;
            break;

        case NPY_VOIDLTR:
            newtype = NPY_VOID;
            break;

        case NPY_DATETIMELTR:
            if (itemsize == 8) {
                newtype = NPY_DATETIME;
            }
            break;

        case NPY_TIMEDELTALTR:
            if (itemsize == 8) {
                newtype = NPY_TIMEDELTA;
            }
            break;
    }

    return newtype;
}

/* Lifted from numarray */
/* TODO: not documented */
/*NUMPY_API
  PyArray_IntTupleFromIntp
*/
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, npy_intp const *vals)
{
    int i;
    PyObject *intTuple = PyTuple_New(len);

    if (!intTuple) {
        goto fail;
    }
    for (i = 0; i < len; i++) {
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
        PyObject *o = PyInt_FromLong((long) vals[i]);
#else
        PyObject *o = PyLong_FromLongLong((npy_longlong) vals[i]);
#endif
        if (!o) {
            Py_DECREF(intTuple);
            intTuple = NULL;
            goto fail;
        }
        PyTuple_SET_ITEM(intTuple, i, o);
    }

 fail:
    return intTuple;
}

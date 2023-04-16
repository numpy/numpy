#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

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


/*
 * Convert the given value to an integer. Replaces the error when compared
 * to `PyArray_PyIntAsIntp`.  Exists mainly to retain old behaviour of
 * `PyArray_IntpConverter` and `PyArray_IntpFromSequence`
 */
static inline npy_intp
dimension_from_scalar(PyObject *ob)
{
    npy_intp value = PyArray_PyIntAsIntp(ob);

    if (error_converting(value)) {
        if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError,
                    "Maximum allowed dimension exceeded");
        }
        return -1;
    }
    return value;
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
    seq->ptr = NULL;
    seq->len = 0;

    /*
     * When the deprecation below expires, remove the `if` statement, and
     * update the comment for PyArray_OptionalIntpConverter.
     */
    if (obj == Py_None) {
        /* Numpy 1.20, 2020-05-31 */
        if (DEPRECATE(
                "Passing None into shape arguments as an alias for () is "
                "deprecated.") < 0){
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }

    PyObject *seq_obj = NULL;

    /*
     * If obj is a scalar we skip all the useless computations and jump to
     * dimension_from_scalar as soon as possible.
     */
    if (!PyLong_CheckExact(obj) && PySequence_Check(obj)) {
        seq_obj = PySequence_Fast(obj,
               "expected a sequence of integers or a single integer.");
        if (seq_obj == NULL) {
            /* continue attempting to parse as a single integer. */
            PyErr_Clear();
        }
    }

    if (seq_obj == NULL) {
        /*
         * obj *might* be a scalar (if dimension_from_scalar does not fail, at
         * the moment no check have been performed to verify this hypothesis).
         */
        seq->ptr = npy_alloc_cache_dim(1);
        if (seq->ptr == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        else {
            seq->len = 1;

            seq->ptr[0] = dimension_from_scalar(obj);
            if (error_converting(seq->ptr[0])) {
                /*
                 * If the error occurred is a type error (cannot convert the
                 * value to an integer) communicate that we expected a sequence
                 * or an integer from the user.
                 */
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_Format(PyExc_TypeError,
                            "expected a sequence of integers or a single "
                            "integer, got '%.100R'", obj);
                }
                npy_free_cache_dim_obj(*seq);
                seq->ptr = NULL;
                return NPY_FAIL;
            }
        }
    }
    else {
        /*
         * `obj` is a sequence converted to the `PySequence_Fast` in `seq_obj`
         */
        Py_ssize_t len = PySequence_Fast_GET_SIZE(seq_obj);
        if (len > NPY_MAXDIMS) {
            PyErr_Format(PyExc_ValueError,
                    "maximum supported dimension for an ndarray "
                    "is %d, found %d", NPY_MAXDIMS, len);
            Py_DECREF(seq_obj);
            return NPY_FAIL;
        }
        if (len > 0) {
            seq->ptr = npy_alloc_cache_dim(len);
            if (seq->ptr == NULL) {
                PyErr_NoMemory();
                Py_DECREF(seq_obj);
                return NPY_FAIL;
            }
        }

        seq->len = len;
        int nd = PyArray_IntpFromIndexSequence(seq_obj,
                (npy_intp *)seq->ptr, len);
        Py_DECREF(seq_obj);

        if (nd == -1 || nd != len) {
            npy_free_cache_dim_obj(*seq);
            seq->ptr = NULL;
            return NPY_FAIL;
        }
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

NPY_NO_EXPORT int
PyArray_CopyConverter(PyObject *obj, _PyArray_CopyMode *copymode) {
    if (obj == Py_None) {
        PyErr_SetString(PyExc_ValueError,
                        "NoneType copy mode not allowed.");
        return NPY_FAIL;
    }

    int int_copymode;
    static PyObject* numpy_CopyMode = NULL;
    npy_cache_import("numpy", "_CopyMode", &numpy_CopyMode);

    if (numpy_CopyMode != NULL && (PyObject *)Py_TYPE(obj) == numpy_CopyMode) {
        PyObject* mode_value = PyObject_GetAttrString(obj, "value");
        if (mode_value == NULL) {
            return NPY_FAIL;
        }

        int_copymode = (int)PyLong_AsLong(mode_value);
        Py_DECREF(mode_value);
        if (error_converting(int_copymode)) {
            return NPY_FAIL;
        }
    }
    else {
        npy_bool bool_copymode;
        if (!PyArray_BoolConverter(obj, &bool_copymode)) {
            return NPY_FAIL;
        }
        int_copymode = (int)bool_copymode;
    }

    *copymode = (_PyArray_CopyMode)int_copymode;
    return NPY_SUCCEED;
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
        if (*axis == NPY_MAXDIMS){
            /* NumPy 1.23, 2022-05-19 */
            if (DEPRECATE("Using `axis=32` (MAXDIMS) is deprecated. "
                          "32/MAXDIMS had the same meaning as `axis=None` which "
                          "should be used instead.  "
                          "(Deprecated NumPy 1.23)") < 0) {
                return NPY_FAIL;
            }
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
        /* str_func returns -1 without an exception if the value is wrong */
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_ValueError,
                "%s %s (got %R)", name, message, object);
        }
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
    else if (str[0] == NPY_BIG || str[0] == NPY_LITTLE ||
             str[0] == NPY_NATIVE || str[0] == NPY_IGNORE) {
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
    int is_exact = 0;

    if (length < 1) {
        return -1;
    }
    else if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
        is_exact = (length == 4 && strcmp(str, "left") == 0);
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
        is_exact = (length == 5 && strcmp(str, "right") == 0);
    }
    else {
        return -1;
    }

    /* Filters out the case sensitive/non-exact
     * match inputs and other inputs and outputs DeprecationWarning
     */
    if (!is_exact) {
        /* NumPy 1.20, 2020-05-19 */
        if (DEPRECATE("inexact matches and case insensitive matches "
                      "for search side are deprecated, please use "
                      "one of 'left' or 'right' instead.") < 0) {
            return -1;
        }
    }

    return 0;
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
    int is_exact = 0;

    if (length < 1) {
        return -1;
    }
    if (str[0] == 'C' || str[0] == 'c') {
        *val = NPY_CLIP;
        is_exact = (length == 4 && strcmp(str, "clip") == 0);
    }
    else if (str[0] == 'W' || str[0] == 'w') {
        *val = NPY_WRAP;
        is_exact = (length == 4 && strcmp(str, "wrap") == 0);
    }
    else if (str[0] == 'R' || str[0] == 'r') {
        *val = NPY_RAISE;
        is_exact = (length == 5 && strcmp(str, "raise") == 0);
    }
    else {
        return -1;
    }

    /* Filters out the case sensitive/non-exact
     * match inputs and other inputs and outputs DeprecationWarning
     */
    if (!is_exact) {
        /* Numpy 1.20, 2020-05-19 */
        if (DEPRECATE("inexact matches and case insensitive matches "
                      "for clip mode are deprecated, please use "
                      "one of 'clip', 'raise', or 'wrap' instead.") < 0) {
            return -1;
        }
    }

    return 0;
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

static int correlatemode_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_CORRELATEMODE *val = (NPY_CORRELATEMODE *)data;
    int is_exact = 0;

    if (length < 1) {
        return -1;
    }
    if (str[0] == 'V' || str[0] == 'v') {
        *val = NPY_VALID;
        is_exact = (length == 5 && strcmp(str, "valid") == 0);
    }
    else if (str[0] == 'S' || str[0] == 's') {
        *val = NPY_SAME;
        is_exact = (length == 4 && strcmp(str, "same") == 0);
    }
    else if (str[0] == 'F' || str[0] == 'f') {
        *val = NPY_FULL;
        is_exact = (length == 4 && strcmp(str, "full") == 0);
    }
    else {
        return -1;
    }

    /* Filters out the case sensitive/non-exact
     * match inputs and other inputs and outputs DeprecationWarning
     */
    if (!is_exact) {
        /* Numpy 1.21, 2021-01-19 */
        if (DEPRECATE("inexact matches and case insensitive matches for "
                      "convolve/correlate mode are deprecated, please "
                      "use one of 'valid', 'same', or 'full' instead.") < 0) {
            return -1;
        }
    }

    return 0;
}

/*
 * Convert an object to NPY_VALID / NPY_SAME / NPY_FULL
 */
NPY_NO_EXPORT int
PyArray_CorrelatemodeConverter(PyObject *object, NPY_CORRELATEMODE *val)
{
    if (PyUnicode_Check(object)) {
        return string_converter_helper(
            object, (void *)val, correlatemode_parser, "mode",
            "must be one of 'valid', 'same', or 'full'");
    }

    else {
        /* For users passing integers */
        int number = PyArray_PyIntAsInt(object);
        if (error_converting(number)) {
            PyErr_SetString(PyExc_TypeError,
                        "convolve/correlate mode not understood");
            return NPY_FAIL;
        }
        if (number <= (int) NPY_FULL
                && number >= (int) NPY_VALID) {
            *val = (NPY_CORRELATEMODE) number;
            return NPY_SUCCEED;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "integer convolve/correlate mode must be 0, 1, or 2");
            return NPY_FAIL;
        }
    }
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


NPY_NO_EXPORT int
PyArray_IntpFromPyIntConverter(PyObject *o, npy_intp *val)
{
    *val = PyArray_PyIntAsIntp(o);
    if (error_converting(*val)) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}


/**
 * Reads values from a sequence of integers and stores them into an array.
 *
 * @param  seq      A sequence created using `PySequence_Fast`.
 * @param  vals     Array used to store dimensions (must be large enough to
 *                      hold `maxvals` values).
 * @param  max_vals Maximum number of dimensions that can be written into `vals`.
 * @return          Number of dimensions or -1 if an error occurred.
 *
 * .. note::
 *
 *   Opposed to PyArray_IntpFromSequence it uses and returns `npy_intp`
 *      for the number of values.
 */
NPY_NO_EXPORT npy_intp
PyArray_IntpFromIndexSequence(PyObject *seq, npy_intp *vals, npy_intp maxvals)
{
    /*
     * First of all, check if sequence is a scalar integer or if it can be
     * "casted" into a scalar.
     */
    Py_ssize_t nd = PySequence_Fast_GET_SIZE(seq);
    PyObject *op;
    for (Py_ssize_t i = 0; i < PyArray_MIN(nd, maxvals); i++) {
        op = PySequence_Fast_GET_ITEM(seq, i);

        vals[i] = dimension_from_scalar(op);
        if (error_converting(vals[i])) {
            return -1;
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
    PyObject *seq_obj = NULL;
    if (!PyLong_CheckExact(seq) && PySequence_Check(seq)) {
        seq_obj = PySequence_Fast(seq,
            "expected a sequence of integers or a single integer");
        if (seq_obj == NULL) {
            /* continue attempting to parse as a single integer. */
            PyErr_Clear();
        }
    }

    if (seq_obj == NULL) {
        vals[0] = dimension_from_scalar(seq);
        if (error_converting(vals[0])) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                        "expected a sequence of integers or a single "
                        "integer, got '%.100R'", seq);
            }
            return -1;
        }
        return 1;
    }
    else {
        int res;
        res = PyArray_IntpFromIndexSequence(seq_obj, vals, (npy_intp)maxvals);
        Py_DECREF(seq_obj);
        return res;
    }
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
        PyObject *o = PyArray_PyIntFromIntp(vals[i]);
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

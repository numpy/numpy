#ifndef _NPY_PRIVATE_CONVERSION_UTILS_H_
#define _NPY_PRIVATE_CONVERSION_UTILS_H_

#include <numpy/ndarraytypes.h>

/**
 * Helper to implement PyArg_Parse* converters. This handles conversion
 * of bytes, and produces appropriate error messages.
 *
 * out:
 *     the pointer to read the argument into
 * str_func:
 *     A function that takes a utf8 string and a length. Outputs the parsed
 *     argument into its last argument, and should return -1 on failure.
 * name:
 *     A name describing this argument, to use in error messages
 * message:
 *     Prepended after `name` with a space when populating ValueErrors. Should
 *     include a description of what values are legal.
 */
NPY_NO_EXPORT int
convert_with_str_parser(
    PyObject *object,
    void *out,
    int (*str_func)(char const*, Py_ssize_t, void*),
    char const *name,
    char const *message);

NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq);

NPY_NO_EXPORT int
PyArray_OptionalIntpConverter(PyObject *obj, PyArray_Dims *seq);

NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf);

NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, npy_bool *val);

NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian);

NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind);

NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr);

NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o);

NPY_NO_EXPORT npy_intp
PyArray_PyIntAsIntp(PyObject *o);

NPY_NO_EXPORT npy_intp
PyArray_IntpFromIndexSequence(PyObject *seq, npy_intp *vals, npy_intp maxvals);

NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, npy_intp *vals, int maxvals);

NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype);

NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, npy_intp const *vals);

NPY_NO_EXPORT int
PyArray_SelectkindConverter(PyObject *obj, NPY_SELECTKIND *selectkind);

/*
 * Converts an axis parameter into an ndim-length C-array of
 * boolean flags, True for each axis specified.
 *
 * If obj is None, everything is set to True. If obj is a tuple,
 * each axis within the tuple is set to True. If obj is an integer,
 * just that axis is set to True.
 */
NPY_NO_EXPORT int
PyArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags);

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
extern NPY_NO_EXPORT int evil_global_disable_warn_O4O8_flag;

#endif

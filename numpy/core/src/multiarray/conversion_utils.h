#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_

#include "numpy/ndarraytypes.h"

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


static NPY_INLINE PyObject *
PyArray_PyIntFromIntp(npy_intp const value)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyLong_FromLong((long)value);
#else
    return PyLong_FromLongLong((npy_longlong)value);
#endif
}

NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, npy_intp const *vals);

NPY_NO_EXPORT int
PyArray_CorrelatemodeConverter(PyObject *object, NPY_CORRELATEMODE *val);

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

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_ */

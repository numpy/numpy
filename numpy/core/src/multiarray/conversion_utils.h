#ifndef _NPY_PRIVATE_CONVERSION_UTILS_H_
#define _NPY_PRIVATE_CONVERSION_UTILS_H_

NPY_NO_EXPORT int
PyArray_Converter(PyObject *object, PyObject **address);

NPY_NO_EXPORT int
PyArray_OutputConverter(PyObject *object, PyArrayObject **address);

NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq);

NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf);

NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, Bool *val);

NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian);

NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind);

NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr);

NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o);

NPY_NO_EXPORT intp
PyArray_PyIntAsIntp(PyObject *o);

NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, intp *vals, int maxvals);

NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype);

NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, intp *vals);

#endif

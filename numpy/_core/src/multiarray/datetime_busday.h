#ifndef NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_

/*
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * This is the 'busday_count' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_count(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * This is the 'is_busday' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_is_busday(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_ */

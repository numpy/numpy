#ifndef NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_

/*
 * This is the Python-exposed datetime_as_string function.
 */
NPY_NO_EXPORT PyObject *
array_datetime_as_string(PyObject *NPY_UNUSED(self), PyObject *args,
                                PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_
#define NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_

NPY_NO_EXPORT void
npy_gil_error(PyObject *type, const char *format, ...);

#endif /* NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_ */

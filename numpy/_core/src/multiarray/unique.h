#ifndef NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_
#define NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_

#ifdef __cplusplus
extern "C" {
#endif

PyObject* array__unique_hash(PyObject *NPY_UNUSED(dummy),
                             PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

#ifdef __cplusplus
}
#endif

#endif  // NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_

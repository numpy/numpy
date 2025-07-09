#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_

extern NPY_NO_EXPORT PyTypeObject PyArrayFunctionDispatcher_Type;

NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
        const char *function_name, PyObject *like,
        PyObject *args, PyObject *kwargs,
        PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames);

NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_ */

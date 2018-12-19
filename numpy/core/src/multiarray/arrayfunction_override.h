#ifndef _NPY_PRIVATE__ARRAYFUNCTION_OVERRIDE_H
#define _NPY_PRIVATE__ARRAYFUNCTION_OVERRIDE_H

NPY_NO_EXPORT PyObject *
array_implement_array_function(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs);

#endif

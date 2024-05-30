#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

typedef struct npy_ma_str_struct {
    PyObject *current_allocator;
    PyObject *array;
    PyObject *array_function;
    PyObject *array_struct;
    PyObject *array_priority;
    PyObject *array_interface;
    PyObject *array_wrap;
    PyObject *array_finalize;
    PyObject *implementation;
    PyObject *axis1;
    PyObject *axis2;
    PyObject *like;
    PyObject *numpy;
    PyObject *where;
    PyObject *convert;
    PyObject *preserve;
    PyObject *convert_if_no_array;
    PyObject *cpu;
    PyObject *dtype;
    PyObject *array_err_msg_substr;
    PyObject *out;
    PyObject *errmode_strings[6];
    PyObject *__dlpack__;
} npy_ma_str_struct;

NPY_VISIBILITY_HIDDEN extern npy_ma_str_struct *npy_ma_str;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

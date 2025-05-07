#ifndef NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_
#define NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_

extern NPY_NO_EXPORT PyGetSetDef array_getsetlist[];
extern NPY_NO_EXPORT int array_descr_set_internal(PyArrayObject *self, PyObject *arg);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_ */

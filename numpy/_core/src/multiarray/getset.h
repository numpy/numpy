#ifndef NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_
#define NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_

extern NPY_NO_EXPORT PyGetSetDef array_getsetlist[];

NPY_NO_EXPORT int array_descr_set_internal(PyArrayObject *self, PyObject *arg);
NPY_NO_EXPORT int array_shape_set_internal(PyArrayObject *self, PyObject *val);

/*
 * Set by PyArray_View to suppress the dtype-setting deprecation
 * in array_descr_set().  See gh-31192.
 */
extern NPY_NO_EXPORT int _numpy_view_dtype_set_in_progress;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_ */

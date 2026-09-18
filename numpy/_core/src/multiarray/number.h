#ifndef NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_

#include "module_state_fields.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    NPY_DECLARE_PYOBJECT_FIELDS(NPY_N_OPS_FIELDS)
} NumericOps;

#ifndef Py_LIMITED_API  /* PyNumberMethods is not in the Limited API */
extern NPY_NO_EXPORT PyNumberMethods array_as_number;
#endif

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_ */

#ifndef _NPY_ARRAY_NUMBER_H_
#define _NPY_ARRAY_NUMBER_H_

extern NPY_NO_EXPORT NumericOps n_ops;
extern NPY_NO_EXPORT PyNumberMethods array_as_number;

NPY_NO_EXPORT int
array_any_nonzero(PyArrayObject *mp);

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void);

#endif
